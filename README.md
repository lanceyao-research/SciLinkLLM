# SciLinkLLM
A framework aimed at bridging experimental observations with computational materials modeling and literature analysis using large language models. For now it is limited to microscopy images.

## Requirements

- Python 3.11+
- ASE (Atomic Simulation Environment)
- OpenCV and Pillow for image processing
- Transformers (https://pypi.org/project/transformers/) and model files (https://huggingface.co/google/gemma-3-12b-it)

## How to use

### For Experiment to Claims Workflow:

1. Run ```example_local.ipynb```
from workflows.microscopy_novelty_workflow import MicroscopyNoveltyAssessmentWorkflow
from workflows.dft_recommendation_workflow import DFTRecommendationsWorkflow
 
 
# Initialize workflow
workflow = MicroscopyNoveltyAssessmentWorkflow(
    google_api_key="",
    futurehouse_api_key="",
    dft_recommendations=False
)
 
# Run complete workflow
result = workflow.run_complete_workflow(
    image_path="data/GO_cafm.tif",
    system_info="data/GO_cafm.json",
)
 
# Get summary
print(workflow.get_summary(result))
 
# Extract the needed data
analysis_text = result["claims_generation"]["full_analysis"]
novel_claims = result["novelty_assessment"]["potentially_novel"]
 
# Initialize DFT workflow
dft_workflow = DFTRecommendationsWorkflow(
    google_api_key="",
    output_dir="dft_results"
)
 
# Generate DFT recommendations
dft_result = dft_workflow.run_from_data(
    analysis_text=analysis_text,
    novel_claims=novel_claims
)
 
print(f"Generated {len(dft_result['recommendations'])} DFT recommendations")
