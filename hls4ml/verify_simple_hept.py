"""Run the HEPT attention kernel through the full hls4ml verification pipeline.

This script takes a trained HEPT attention-block checkpoint (see
train_simple_hept_qat.py) and walks it through every verification stage hls4ml
and Vivado offer, from the fastest/cheapest ("does the fixed-point math match
PyTorch?") all the way down to the slowest/most expensive ("does this design
actually fit and route on a real FPGA?"). Every stage after the first is OFF by
default -- turn on only the ones you need with the switches below, since each
extra stage costs meaningfully more wall-clock time than the last.

THE PIPELINE, FASTEST TO SLOWEST
---------------------------------
1. Fast functional check (always runs, unless --skip-fast-check)
   hls4ml compiles the generated C++ kernel into a Python-callable shared
   library and runs it directly -- no Vitis HLS project, no waiting. This is
   the same bit-accurate fixed-point math your final design will use, so if
   the correlation against the PyTorch reference looks wrong here, every
   later stage will just waste time confirming the same bug. Seconds to ~1
   minute.

2. --csynth : Vitis HLS C synthesis
   Vitis HLS reads the C++ kernel and its #pragma HLS directives and
   translates it into an RTL (Verilog) design, scheduling every operation
   into clock cycles. This is where you first get DSP/LUT/FF/BRAM resource
   estimates and a latency/II estimate -- but they are *estimates* based on
   scheduling, not on an actual chip layout. Several minutes to ~1 hour for
   a design this size.

3. --cosim : RTL co-simulation (needs csynth's RTL; implies --csynth)
   Runs your original C testbench against the RTL that csynth just
   generated, inside a cycle-accurate RTL simulator. This checks that Vitis
   HLS's C-to-RTL translation didn't introduce a bug -- a different question
   from "is the fixed-point math correct?" (that's what the fast check in
   step 1 already answers). Much slower than step 1: potentially tens of
   minutes, since it simulates every clock cycle of real hardware behavior.

4. --vsynth : quick Vivado logic-synthesis estimate (needs csynth's RTL; implies --csynth)
   A quick, synthesis-only pass through Vivado (synth_design + some generic
   logic optimization) against the FPGA part named in your hls4ml config.
   This confirms the RTL actually maps onto real FPGA primitives and
   refines the resource estimate a bit -- but it does NOT place or route
   anything, so it cannot tell you whether the design will physically fit
   or whether routing will succeed. Usually a few minutes.

5. --fifo-opt : FIFO depth optimization (needs csynth; implies --csynth)
   The hls::stream buffers hls4ml inserts between dataflow stages default to
   a conservative, fixed depth. This pass runs an internal cosim to profile
   how full each FIFO actually gets, then reports the minimum safe depth --
   letting you shrink oversized FIFOs (saving BRAM/LUT) by hand afterwards.
   Adds a cosim-like amount of time to the csynth step.

6. --export-ip : export a reusable Vivado IP-XACT core (needs csynth's RTL; implies --csynth)
   Packages the synthesized design as a standalone IP block you could drop
   into a larger Vivado block design elsewhere.

7. --place-and-route : full physical implementation (needs csynth's RTL; implies --csynth)
   THE step that actually matters for "will this run on my board?". Takes
   the synthesized RTL and does what Vitis HLS/vsynth never do: lays real
   LUTs/FFs/DSPs/BRAMs out on a specific physical chip, wires them together
   with real routing resources, and reports whether it actually succeeded --
   plus true final utilization and timing (WNS/TNS). This is the ONLY stage
   that can catch routing congestion or cross-die (SLR) routability problems
   that are completely invisible at the csynth/vsynth level: a HEPT
   configuration can look fine in every earlier stage (low utilization,
   clean csynth report) and still fail to fully route once placed on a real,
   multi-die part. By far the slowest stage: tens of minutes to several
   hours, depending on how close the design sits to the device's capacity.

Each stage after the first requires the one before it: --cosim/--vsynth/
--fifo-opt/--export-ip/--place-and-route all need the RTL that csynth
produces, so passing any of them automatically also runs csynth (you don't
need to pass --csynth yourself, though it doesn't hurt to).

ENVIRONMENT
-----------
Before running anything beyond the fast check, you need Xilinx's tools (and
this repository's Python package) on your PATH/PYTHONPATH in the SAME shell
you invoke this script from:

    source /path/to/Xilinx/Vitis_HLS/2024.1/settings64.sh   # for --csynth and friends
    source /path/to/Xilinx/Vivado/2024.1/settings64.sh      # for --place-and-route
    export PYTHONPATH=/path/to/this/repo                    # the directory containing
                                                              # the "hls4ml" package, i.e.
                                                              # the parent of this script's
                                                              # own directory

(If you have a different hls4ml pip-installed elsewhere, PYTHONPATH must come
first on the path for `import hls4ml` to pick up *this* checkout instead --
this script prints which one it loaded on startup so you can double check.)

EXAMPLES
--------
Fast correctness check only (default; no Xilinx tools needed):
    python verify_simple_hept.py --checkpoint path/to/hept_block0.pt --real-data

Add C synthesis to see resource/latency estimates:
    python verify_simple_hept.py --checkpoint path/to/hept_block0.pt --real-data --csynth

Go all the way to a real place-and-route run on an Alveo U250:
    python verify_simple_hept.py --checkpoint path/to/hept_block0.pt --real-data \\
        --place-and-route --output-dir hept_test_my_run
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import convertSimpleHEPT


def check_tool_on_path(tool, hint):
    """Give a clear, actionable error instead of a cryptic 'command not found' if a
    required Xilinx tool isn't on PATH in this shell -- a very common first-time
    stumbling block (the Vitis/Vivado tools are never on PATH by default; you have
    to `source .../settings64.sh` yourself in every new shell)."""
    if shutil.which(tool) is None:
        raise SystemExit(
            f"'{tool}' was not found on PATH.\n"
            f"Hint: {hint}\n"
            "Remember this only lasts for the current shell session -- you'll need to "
            "source it again next time you open a new terminal."
        )


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def generate_place_and_route_tcl(project_name, verilog_dir, clock_period, part,
                                  place_directive, phys_opt_directive, route_directive):
    """Build the Tcl script for a full, non-project-mode Vivado implementation run.

    This is deliberately NOT the same thing as --vsynth above: vsynth only runs
    synth_design (logic synthesis), while this runs the complete
    synth -> opt -> place -> phys_opt -> route flow against a real part, which is
    the only way to find out whether the design truly fits and routes.

    "Out-of-context" mode (synth_design -mode out_of_context) treats the kernel as
    a standalone block without a surrounding top-level design or I/O buffers --
    appropriate here since we only care about the kernel's own resource usage and
    routability, not a full board design.
    """
    return f"""\
# Auto-generated by verify_simple_hept.py -- full Vivado place-and-route flow.
# Every stage below writes its own checkpoint (.dcp), so if a later stage fails
# you can re-open an earlier checkpoint and retry just that stage instead of
# starting over from scratch (`open_checkpoint {project_name}_post_place.dcp`
# followed by whichever Tcl commands you want to re-run, for example).

# Vitis HLS's own synthesis target part (set in your hls4ml config, e.g.
# xcvu13p-flga2577-2-e) is NOT necessarily the part you actually have a Vivado
# license/board for. This flow explicitly re-targets a real, licensed part --
# pass --part on the command line to point this at your own board instead.
set part "{part}"

create_project -force -in_memory impl_proj
set_property part $part [current_project]

# Pull in the Verilog Vitis HLS already generated during --csynth -- this step
# does not re-run HLS, it only re-targets the existing RTL at a real chip.
add_files {verilog_dir}
set_property top {project_name} [current_fileset]

# A minimal clock constraint is required before synth_design will do anything
# useful with timing. {clock_period} ns matches the ClockPeriod this project
# was built with; pass --clock-period-override to try a looser constraint if
# you suspect the design is routable but just can't meet this particular
# timing budget.
set xdc_fh [open {project_name}_clock.xdc w]
puts $xdc_fh "create_clock -name ap_clk -period {clock_period} \\[get_ports ap_clk\\]"
close $xdc_fh
add_files -fileset constrs_1 {project_name}_clock.xdc

puts "\\n***** SYNTH_DESIGN (logic synthesis) *****"
set t0 [clock clicks -milliseconds]
synth_design -top {project_name} -part $part -mode out_of_context
puts "SYNTH_DESIGN: [expr {{([clock clicks -milliseconds]-$t0)/1000.0}}] s"
write_checkpoint -force {project_name}_post_synth.dcp
report_utilization -file {project_name}_utilization_synth.rpt

puts "\\n***** OPT_DESIGN (pre-placement netlist optimization) *****"
set t0 [clock clicks -milliseconds]
opt_design
puts "OPT_DESIGN: [expr {{([clock clicks -milliseconds]-$t0)/1000.0}}] s"
write_checkpoint -force {project_name}_post_opt.dcp

# -directive {place_directive}: on multi-die ("SLR"/stacked-silicon) parts like the
# Alveo U250, the SSI_* placement directives explicitly try to spread logic more
# evenly across die boundaries, which plain default placement often doesn't do on
# its own -- see this project's notes on num_heads=1 P&R for a worked example of
# why this matters. Try a different directive (e.g. SSI_SpreadLogic_high vs.
# SSI_BalanceSLLs) if this one performs poorly on your design; be aware results
# can differ by orders of magnitude between directives in a way that is NOT
# obviously predictable from utilization numbers alone.
puts "\\n***** PLACE_DESIGN (directive: {place_directive}) *****"
set t0 [clock clicks -milliseconds]
place_design -directive {place_directive}
puts "PLACE_DESIGN: [expr {{([clock clicks -milliseconds]-$t0)/1000.0}}] s"
write_checkpoint -force {project_name}_post_place.dcp
report_utilization -file {project_name}_utilization_placed.rpt

puts "\\n***** PHYS_OPT_DESIGN (directive: {phys_opt_directive}) *****"
set t0 [clock clicks -milliseconds]
phys_opt_design -directive {phys_opt_directive}
puts "PHYS_OPT_DESIGN: [expr {{([clock clicks -milliseconds]-$t0)/1000.0}}] s"
write_checkpoint -force {project_name}_post_physopt.dcp

puts "\\n***** ROUTE_DESIGN (directive: {route_directive}) *****"
set t0 [clock clicks -milliseconds]
route_design -directive {route_directive}
puts "ROUTE_DESIGN: [expr {{([clock clicks -milliseconds]-$t0)/1000.0}}] s"
write_checkpoint -force {project_name}_post_route.dcp

report_utilization -file {project_name}_utilization_routed.rpt
report_timing_summary -file {project_name}_timing_routed.rpt -delay_type min_max
report_design_analysis -congestion -file {project_name}_congestion.rpt

puts "\\n***** PLACE-AND-ROUTE COMPLETE *****"
puts "Look for these lines in the log / report_route_status output to judge success:"
puts "  'Number of Failed Nets' / 'Number of Node Overlaps' -- 0 for a clean route;"
puts "  any unroutable-pins CRITICAL WARNING means the design did NOT fully route"
puts "  even if route_design reports 'completed successfully'."
exit
"""


def run_place_and_route(hls_model, args):
    """Run the full Vivado place-and-route flow in its own subdirectory of the
    hls4ml output directory, using the RTL --csynth already produced."""
    check_tool_on_path(
        "vivado",
        "source <your Xilinx install>/Vivado/<version>/settings64.sh in this shell.",
    )

    project_name = hls_model.config.get_project_name()
    project_dir = hls_model.config.get_project_dir()
    output_dir = Path(hls_model.config.get_output_dir())
    verilog_dir = output_dir / project_dir / "solution1" / "syn" / "verilog"
    if not verilog_dir.is_dir():
        raise SystemExit(
            f"No synthesized RTL found at {verilog_dir}.\n"
            "This shouldn't happen -- csynth should have run automatically before "
            "place-and-route. Try re-running with --csynth explicitly."
        )

    clock_period = args.clock_period_override or hls_model.config.get_config_value("ClockPeriod")

    pr_dir = output_dir / "place_and_route"
    pr_dir.mkdir(exist_ok=True)
    # The Tcl script lives in its own subdirectory, so it needs a path back up to
    # the actual HLS project's generated Verilog.
    relative_verilog_dir = Path("..") / project_dir / "solution1" / "syn" / "verilog"

    tcl_path = pr_dir / "place_and_route.tcl"
    tcl_path.write_text(
        generate_place_and_route_tcl(
            project_name=project_name,
            verilog_dir=relative_verilog_dir.as_posix(),
            clock_period=clock_period,
            part=args.part,
            place_directive=args.place_directive,
            phys_opt_directive=args.phys_opt_directive,
            route_directive=args.route_directive,
        )
    )

    banner(
        f"PLACE-AND-ROUTE -- targeting {args.part}\n"
        f"This is the slowest stage in the pipeline: expect anywhere from ~20 minutes "
        f"to several hours. Vivado's progress will print below as it runs."
    )
    print(f"Generated Tcl script: {tcl_path}")
    print(f"Working directory   : {pr_dir}")

    t0 = time.time()
    result = subprocess.run(
        ["vivado", "-mode", "batch", "-source", tcl_path.name],
        cwd=pr_dir,
    )
    elapsed_min = (time.time() - t0) / 60.0
    if result.returncode != 0:
        raise SystemExit(
            f"Place-and-route failed after {elapsed_min:.1f} minutes (Vivado exit code "
            f"{result.returncode}). Scroll up for the specific ERROR/CRITICAL WARNING "
            f"lines, or re-read {pr_dir / 'vivado.log'}."
        )

    print(f"\nPlace-and-route finished in {elapsed_min:.1f} minutes.")
    print(f"Reports written to: {pr_dir}")
    print(f"  {project_name}_utilization_placed.rpt  -- utilization right after placement")
    print(f"  {project_name}_utilization_routed.rpt  -- final utilization after routing")
    print(f"  {project_name}_timing_routed.rpt        -- final timing (WNS/TNS)")
    print(f"  {project_name}_congestion.rpt           -- routing congestion hotspots, if any")


def build_arg_parser():
    # Start from convertSimpleHEPT's parser so every architecture/precision/checkpoint
    # flag (--checkpoint, --real-data, --num-heads, --hgq-precision, ...) stays in sync
    # with that script automatically -- see its module docstring for what those do.
    parser = convertSimpleHEPT.build_arg_parser()
    parser.description = (
        "Build a HEPT attention HLS project from a checkpoint and run it through as "
        "much of the hls4ml/Vivado verification pipeline (csynth, cosim, vsynth, "
        "fifo_opt, IP export, full place-and-route) as requested. See the top of this "
        "file for a detailed explanation of each stage."
    )

    pipeline = parser.add_argument_group("pipeline stage switches (each one after the first turns on more, slower, checking)")
    pipeline.add_argument(
        "--skip-fast-check", action="store_true",
        help="Skip the quick Python-side (hls4ml predict()) correctness check that "
        "normally always runs first. Only useful if you've already confirmed "
        "correctness and just want to re-run a later stage faster.",
    )
    pipeline.add_argument(
        "--csynth", action="store_true",
        help="Run Vitis HLS C synthesis: translates the C++ kernel into RTL and "
        "produces first resource/latency estimates. Required by (and automatically "
        "enabled by) --cosim/--vsynth/--fifo-opt/--export-ip/--place-and-route.",
    )
    pipeline.add_argument(
        "--cosim", action="store_true",
        help="Run RTL co-simulation: replays your C testbench against the synthesized "
        "RTL in a cycle-accurate simulator, to catch bugs in Vitis HLS's C-to-RTL "
        "translation itself. Slower than the fast check. Implies --csynth.",
    )
    pipeline.add_argument(
        "--vsynth", action="store_true",
        help="Run a quick logic-synthesis-only pass through Vivado (no placement, no "
        "routing) to sanity-check the RTL maps onto real FPGA primitives. Implies "
        "--csynth. NOT a substitute for --place-and-route: it cannot tell you "
        "whether the design will actually fit or route.",
    )
    pipeline.add_argument(
        "--fifo-opt", action="store_true",
        help="Let Vitis HLS profile actual hls::stream FIFO occupancy (via an internal "
        "cosim run) and report the minimum safe depth for each one, so oversized "
        "default FIFO depths can be trimmed by hand afterwards. Implies --csynth.",
    )
    pipeline.add_argument(
        "--export-ip", action="store_true",
        help="Export the synthesized design as a standalone Vivado IP-XACT core. "
        "Implies --csynth.",
    )
    pipeline.add_argument(
        "--place-and-route", action="store_true",
        help="Run the full, real Vivado implementation flow (synth -> opt -> place -> "
        "phys_opt -> route) against an actual FPGA part. This is the only stage that "
        "can tell you whether the design truly fits and routes -- see the module "
        "docstring for why csynth/vsynth alone can't answer that. By far the slowest "
        "stage (tens of minutes to several hours). Implies --csynth.",
    )

    pr = parser.add_argument_group("place-and-route options (only used with --place-and-route)")
    pr.add_argument(
        "--part", type=str, default="xcu250-figd2104-2L-e",
        help="Target FPGA part for --place-and-route (default: an Alveo U250, which is "
        "what this project's own investigations used). This is independent of whatever "
        "part your hls4ml config targets for csynth/vsynth -- set it to whatever "
        "board/part you actually have a Vivado license for.",
    )
    pr.add_argument(
        "--place-directive", type=str, default="SSI_SpreadLogic_high",
        help="Vivado place_design -directive (default: SSI_SpreadLogic_high, which "
        "spreads logic across a multi-die part's SLR boundaries more aggressively "
        "than default placement). Other options worth trying if this one performs "
        "poorly: SSI_BalanceSLLs, SSI_SpreadSLLs, or the Vivado default (pass "
        "'Default'). Results can vary by orders of magnitude between directives in "
        "ways that aren't obviously predictable from utilization alone -- if one "
        "directive performs badly, that does not mean a nearby one will too, or that "
        "it won't be dramatically worse.",
    )
    pr.add_argument(
        "--phys-opt-directive", type=str, default="AggressiveExplore",
        help="Vivado phys_opt_design -directive (default: AggressiveExplore).",
    )
    pr.add_argument(
        "--route-directive", type=str, default="AggressiveExplore",
        help="Vivado route_design -directive (default: AggressiveExplore).",
    )
    pr.add_argument(
        "--clock-period-override", type=float, default=None,
        help="Use a different clock period (in ns) for place-and-route than the one "
        "this HLS project was built with (hls_config['ClockPeriod'], default 5ns). "
        "Relaxing this can sometimes turn a routing failure into a success at the "
        "cost of a slower clock -- useful to try if a design is close to routing but "
        "not quite making it.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    import hls4ml
    print(f"Using hls4ml from: {hls4ml.__file__}")
    print("(If that's not the checkout you expected, fix PYTHONPATH and re-run --"
          " see this script's module docstring.)")

    # --cosim/--vsynth/--fifo-opt/--export-ip/--place-and-route all consume the RTL
    # that csynth produces, so asking for any of them implies csynth too.
    need_rtl = args.cosim or args.vsynth or args.fifo_opt or args.export_ip or args.place_and_route
    run_csynth = args.csynth or need_rtl

    banner("STEP 1/3: building the PyTorch reference and converting to an hls4ml project")
    hls_model, reference, x, coords, combined_shifts = convertSimpleHEPT.build_hls_project(args)

    if not args.skip_fast_check:
        banner("STEP 2/3: fast functional check (hls4ml predict() vs. PyTorch reference)")
        hls_model.compile()
        hls_prediction = np.asarray(hls_model.predict([x, coords, combined_shifts])).reshape(reference.shape)
        convertSimpleHEPT.report(reference, hls_prediction)
    else:
        print("\n(skipping the fast functional check, per --skip-fast-check)")

    if run_csynth:
        banner(
            "STEP 3/3: Vitis HLS build "
            f"(csynth=True, cosim={args.cosim}, vsynth={args.vsynth}, "
            f"fifo_opt={args.fifo_opt}, export={args.export_ip})"
        )
        check_tool_on_path(
            "vitis-run",
            "source <your Xilinx install>/Vitis_HLS/<version>/settings64.sh in this shell.",
        )
        report = hls_model.build(
            reset=False,
            csim=False,  # the fast check above already validated the fixed-point math;
                         # this would re-check it the slow way, inside Vitis HLS itself.
            synth=True,
            cosim=args.cosim,
            validation=False,
            export=args.export_ip,
            vsynth=args.vsynth,
            fifo_opt=args.fifo_opt,
        )
        print("\nVitis HLS build report:")
        print(report)
    else:
        print(
            "\n(no --csynth/--cosim/--vsynth/--fifo-opt/--export-ip/--place-and-route "
            "requested -- stopping after the fast functional check)"
        )

    if args.place_and_route:
        run_place_and_route(hls_model, args)

    banner("DONE")
    print(f"All project files are under: {hls_model.config.get_output_dir()}")


if __name__ == "__main__":
    main()
