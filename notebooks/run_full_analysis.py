#!/usr/bin/env python3
"""
Complete SUPERDIFF Analysis with Enhanced Visualizations

Runs:
1. Spatial grounding experiments (semantic vs. spatial)
2. Enhanced visualizations (unified 2D/3D, trajectories, temporal)
3. Generates comprehensive report

Usage:
    python notebooks/run_full_analysis.py
    python notebooks/run_full_analysis.py --quick
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.spatial_grounding_experiments import (
    SpatialGroundingConfig,
    SpatialGroundingExperimentSuite
)
from notebooks.enhanced_visualizations import generate_enhanced_visualizations


def main():
    parser = argparse.ArgumentParser(
        description='Complete SUPERDIFF analysis with enhanced visualizations'
    )

    # Prompts
    parser.add_argument('--object-a', type=str, default="a cat")
    parser.add_argument('--object-b', type=str, default="a dog")
    parser.add_argument('--spatial-desc-a', type=str, default="on the left side")
    parser.add_argument('--spatial-desc-b', type=str, default="on the right side")

    # Parameters
    parser.add_argument('--num-runs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--lift', type=float, default=0.0)

    # Options
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-enhanced-viz', action='store_true',
                       help='Skip enhanced visualizations (faster)')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.quick:
        args.num_runs = 5
        args.steps = 100
        print("\n[QUICK MODE] Reduced parameters for faster testing\n")

    # Construct prompts
    semantic_a = args.object_a
    semantic_b = args.object_b
    semantic_composed = f"{args.object_a} and {args.object_b}"

    spatial_a = f"{args.object_a} {args.spatial_desc_a}"
    spatial_b = f"{args.object_b} {args.spatial_desc_b}"
    spatial_composed = f"{args.object_a} {args.spatial_desc_a} and {args.object_b} {args.spatial_desc_b}"

    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/full_analysis_{timestamp}"

    config = SpatialGroundingConfig(
        semantic_a=semantic_a,
        semantic_b=semantic_b,
        semantic_composed=semantic_composed,
        spatial_a=spatial_a,
        spatial_b=spatial_b,
        spatial_composed=spatial_composed,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        lift=args.lift,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("COMPLETE SUPERDIFF ANALYSIS")
    print("="*80)
    print("\nThis runs:")
    print("  1. Spatial grounding experiments (semantic vs. spatial)")
    print("  2. Enhanced visualizations (unified 2D/3D, trajectories, temporal)")
    print("  3. Comprehensive analysis report")
    print("\n" + "="*80)
    print("\nConfiguration:")
    print(f"\n  SEMANTIC:")
    print(f"    {config.semantic_a} and {config.semantic_b}")
    print(f"  SPATIAL:")
    print(f"    {config.spatial_a} and {config.spatial_b}")
    print(f"\n  Runs: {config.num_runs}")
    print(f"  Steps: {config.num_inference_steps}")
    print(f"  Output: {config.output_dir}")
    print("="*80 + "\n")

    # Step 1: Run spatial grounding experiments
    print("\n" + "="*80)
    print("STEP 1: SPATIAL GROUNDING EXPERIMENTS")
    print("="*80 + "\n")

    suite = SpatialGroundingExperimentSuite(config)
    suite.run_all_experiments()

    # Step 2: Generate enhanced visualizations
    if not args.skip_enhanced_viz:
        print("\n" + "="*80)
        print("STEP 2: ENHANCED VISUALIZATIONS")
        print("="*80)

        # Enhanced viz for semantic prompts
        print("\n--- Semantic Prompts ---")
        generate_enhanced_visualizations(
            suite.semantic_suite.results,
            str(Path(config.output_dir) / "semantic" / "enhanced"),
            suite.semantic_suite.config
        )

        # Enhanced viz for spatial prompts
        print("\n--- Spatial Prompts ---")
        generate_enhanced_visualizations(
            suite.spatial_suite.results,
            str(Path(config.output_dir) / "spatial" / "enhanced"),
            suite.spatial_suite.config
        )

    # Step 3: Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    output_path = Path(config.output_dir)

    print(f"\nResults saved to: {output_path.absolute()}")
    print("\n" + "="*80)
    print("KEY FILES TO REVIEW")
    print("="*80)

    print("\n1. CRITICAL VISUAL INSPECTION:")
    print(f"   {output_path / 'semantic_vs_spatial_comparison.png'}")
    print("   → Compare Row 2 (semantic SUPERDIFF) vs Row 4 (spatial SUPERDIFF)")
    print("   → Does spatial grounding produce co-presence or still hybrid?")

    print("\n2. COMPREHENSIVE INTERPRETATION:")
    print(f"   {output_path / 'comparative_report.txt'}")
    print("   → Full analysis with automatic interpretation")

    print("\n3. UNIFIED LATENT SPACE:")
    print(f"   {output_path / 'semantic/enhanced/unified_latent_space_2d.png'}")
    print(f"   {output_path / 'semantic/enhanced/unified_latent_space_3d_interactive.html'}")
    print("   → All conditions in single plot")
    print("   → Check: Is SUPERDIFF on A-B line?")

    print("\n4. TRAJECTORY DYNAMICS:")
    print(f"   {output_path / 'semantic/enhanced/trajectory_evolution_3d_run0_sample0.png'}")
    print(f"   {output_path / 'semantic/enhanced/trajectory_evolution_3d_interactive_run0_sample0.html'}")
    print("   → Path evolution through latent space")
    print("   → Check: Smooth (on-manifold) or sharp turns (off-manifold)?")

    print("\n5. TEMPORAL ANALYSIS:")
    print(f"   {output_path / 'semantic/enhanced/temporal_phase_diagram.png'}")
    print("   → When does divergence occur? (early/mid/late)")
    print("   → Critical timesteps for composition")

    print("\n" + "="*80)
    print("INTERPRETATION DECISION TREE")
    print("="*80)

    print("""
1. Visual Inspection (semantic_vs_spatial_comparison.png):

   IF Row 4 (spatial) shows co-presence:
     → Issue is LACK OF SPATIAL INDUCTIVE BIAS
     → Solution: Use spatial prompts
     → Hybridization is NOT fundamental

   ELSE IF Row 4 still shows hybrids:
     → Continue to geometric analysis...

2. Geometric Analysis (unified_latent_space_2d.png):

   IF SUPERDIFF centroid near (A+B)/2 midpoint:
     → Linear interpolation
     → Check manifold adherence...

     IF Geodesic ≈ Euclidean (trajectory smooth):
       → INTRINSIC to probability product
       → Need different composition operator

     ELSE IF Geodesic >> Euclidean (sharp turns):
       → GEOMETRIC ARTIFACT (off-manifold)
       → Fixable with geodesic interpolation

   ELSE IF SUPERDIFF centroid near monolithic:
     → Composition discovers natural semantics
     → Mathematical AND ≈ linguistic "and"

3. Temporal Analysis (temporal_phase_diagram.png):

   → Identify WHEN composition mechanism engages
   → Analyze velocity fields at critical timesteps
   → Understanding temporal dynamics informs mechanism
""")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    print("""
1. Open semantic_vs_spatial_comparison.png
   → This SINGLE IMAGE answers the spatial grounding hypothesis

2. Read comparative_report.txt
   → Automated interpretation of all metrics

3. Explore interactive 3D plots (*.html files)
   → Rotate to see geometric relationships
   → Verify 2D projection interpretations

4. Based on findings, run parameter sweeps:
   - Vary lift parameter
   - Test different spatial descriptions
   - Try alternative prompt types
""")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
