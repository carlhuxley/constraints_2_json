#!/usr/bin/env python3
"""
Run ACE TDD Agent for constraints_2_json project.
Uses file mode (no PostgreSQL required).
"""
import sys
sys.path.insert(0, "/home/ch_dev/ace_enterprise")

import logging
from pathlib import Path

from src.agents.autonomous_tdd_agent import AutonomousTDDAgent
from src.agents.test_review_agent import TestReviewAgent
from src.ensemble.learner import EnsembleLearner
from src.config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 80)
    print("  ACE TDD AGENT - JSON Schema Constraint Enrichment")
    print("=" * 80)

    # Setup workspace
    project_root = Path("/home/ch_dev/constraints_2_json")
    test_dir = project_root / "tests"
    src_dir = project_root / "src"

    test_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py for package
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    print(f"\n📁 Project: {project_root}")
    print(f"   Source: {src_dir}")
    print(f"   Tests: {test_dir}")

    # Use DeepSeek via OpenRouter (open-source, allowed by ACE)
    provider = "openrouter"
    model = "deepseek/deepseek-chat"

    print(f"\n⚙️  Using model: {provider}/{model}")

    # Create ensemble learner in file mode (no PostgreSQL)
    ensemble = EnsembleLearner(
        models=[(provider, model)],
        playbook_id="constraints_2_json_playbook",
        enable_deliberation=False,
    )

    # Disable playbook manager to avoid PostgreSQL requirement
    ensemble.playbook_manager = None

    # Test reviewer
    test_reviewer = TestReviewAgent(use_llm_analysis=False)

    # TDD Agent
    agent = AutonomousTDDAgent(
        ensemble_learner=ensemble,
        test_reviewer=test_reviewer,
        project_root=project_root,
        test_dir=test_dir,
        src_dir=src_dir,
        max_iterations=15,
        review_threshold=0.7
    )

    # Disable components that require PostgreSQL
    agent.playbook_manager = None
    agent.bullet_retriever = None

    print("✅ Components initialized (file mode - no PostgreSQL)")

    # Read the feature file
    feature_file = project_root / "features" / "schema_enrichment.feature"
    if not feature_file.exists():
        print(f"❌ Feature file not found: {feature_file}")
        return 1

    feature_content = feature_file.read_text()

    print(f"\n🔨 Building feature from: {feature_file.name}")
    print("=" * 80)

    try:
        # Build the feature
        result = agent.build_feature(
            requirement=feature_content,
            gherkin_dir=feature_file.parent,
            project_root=project_root,
            source_dir=src_dir,
            test_dir=test_dir,
        )

        print("\n" + "=" * 80)
        print("✅ FEATURE COMPLETE!")
        print("=" * 80)
        print(f"  • Cycles executed: {result.cycles_executed}")
        print(f"  • Tests created: {len(result.test_files)}")
        print(f"  • Implementation files: {len(result.implementation_files)}")
        print(f"  • All tests passed: {result.all_tests_passed}")
        print(f"  • Time: {result.total_time_seconds:.1f}s")

        print("\n📄 Generated Files:")
        for f in result.test_files + result.implementation_files:
            if f.exists():
                print(f"  • {f.relative_to(project_root)}")

        return 0

    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
