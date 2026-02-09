#!/usr/bin/env python3
"""
Example: Multi-Prompt SuperDiff Composition

This script demonstrates how to compose 3 or more prompts using the SuperDiff method.
It compares the individual prompts with the composed result.

Usage:
    python notebooks/example_multi_composition.py
    python notebooks/example_multi_composition.py --prompts "a cat" "a dog" "a bird" "all three animals together"
"""

import argparse
from manifold_diagnostics import ManifoldDiagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-prompt SuperDiff composition example"
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+",
        default=["a cat", "a dog", "a cat and a dog"],
        help="Prompts to compose (3 or more recommended)"
    )
    parser.add_argument(
        "--operation", type=str, default="AND",
        choices=["AND", "OR"],
        help="Composition operation: AND (joint satisfaction) or OR (softmax selection)"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--lift", type=float, default=0.0,
        help="Lift parameter for stability (try 10.0-100.0 for stronger effects)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/multi_composition_example",
        help="Output directory for results"
    )

    args = parser.parse_args()

    print("="*80)
    print("MULTI-PROMPT SUPERDIFF COMPOSITION EXAMPLE")
    print("="*80)
    print(f"\nPrompts to compose: {args.prompts}")
    print(f"Operation: {args.operation}")
    print(f"Steps: {args.steps}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"Lift: {args.lift}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    # Initialize diagnostics toolkit
    diag = ManifoldDiagnostics()

    # Run multi-prompt composition
    results = diag.multi_prompt_composition(
        prompts=args.prompts,
        operation=args.operation,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        lift=args.lift,
        output_dir=args.output_dir,
    )

    # Print summary statistics
    print("\n" + "="*80)
    print("COMPOSITION SUMMARY")
    print("="*80)

    final_kappas = results["kappas"][-1]  # (batch, M)
    print("\nFinal composition weights (kappa) averaged over batch:")
    for i, prompt in enumerate(args.prompts):
        mean_kappa = final_kappas[:, i].mean()
        std_kappa = final_kappas[:, i].std()
        print(f"  {i+1}. '{prompt}': κ = {mean_kappa:.4f} ± {std_kappa:.4f}")

    final_ll = results["log_likelihoods"][-1]  # (batch, M)
    print("\nFinal log-likelihoods averaged over batch:")
    for i, prompt in enumerate(args.prompts):
        mean_ll = final_ll[:, i].mean()
        std_ll = final_ll[:, i].std()
        print(f"  {i+1}. '{prompt}': ℓ = {mean_ll:.2f} ± {std_ll:.2f}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Higher κ (kappa) values indicate stronger contribution to the final composition.
- If κ values are similar, all prompts contribute equally.
- If one κ dominates, that concept is most prominent in the output.

For AND operation:
- All concepts should be present, κ values show relative importance.

For OR operation:
- Higher κ indicates the selected concept(s) via softmax.
- Usually one prompt will have dominant κ ≈ 1.

The lift parameter can adjust the balance:
- Higher lift encourages more uniform κ distribution.
- Lower (or negative) lift allows sharper selection.
""")

    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
