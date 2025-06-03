# FILE: exp_agents/microscopy_agent.py

import json
import os
from io import BytesIO
from PIL import Image
import logging
import numpy as np

# Keep original instructions
from .instruct import MICROSCOPY_ANALYSIS_INSTRUCTIONS, MICROSCOPY_CLAIMS_INSTRUCTIONS, FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS, DIFFRACTION_ANALYSIS_INSTRUCTIONS
from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes, normalize_and_convert_to_image_bytes, normalize_and_convert_to_PIL
from .fft_nmf_analyzer import SlidingFFTNMF

# Import AI package
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

class LocalMicroscopyAnalysisAgent:
    """
    Agent for analyzing microscopy images using Local Gemma3-12B models.
    Refactored to reduce code duplication between analysis methods.
    """

    def __init__(self, model_name: str = "gemma3_12B_local", fft_nmf_settings: dict | None = None):
        self.model = Gemma3ForConditionalGeneration.from_pretrained("../" + model_name, torch_dtype=torch.bfloat16).eval().to("cuda")
        self.processor = AutoProcessor.from_pretrained("../" + model_name)
        self.logger = logging.getLogger(__name__)
        self.fft_nmf_settings = fft_nmf_settings if fft_nmf_settings else {} # Ensure it's a dict
        self.RUN_FFT_NMF = self.fft_nmf_settings.get('FFT_NMF_ENABLED', False)
        self.FFT_NMF_AUTO_PARAMS = self.fft_nmf_settings.get('FFT_NMF_AUTO_PARAMS', False)

    def _analyze_diffraction_base(self, image_path: str, system_info: dict | str | None, instruction_prompt: str) -> tuple[dict | None, dict | None]:
        """
        Internal helper method to handle common image analysis steps.
        """
        try:
            
            loaded_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(loaded_image)
            image = Image.fromarray(preprocessed_img_array)

            messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction_prompt}]
            }]
            user_messages = []
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            user_messages.append({"type": "text", "text": analysis_request_text})
            user_messages.append({"type": "image", "image": image})

            if system_info:
                system_info_text = "\n\nAdditional System Information:\n"
                if isinstance(system_info, str):
                    try: system_info_text += json.dumps(json.loads(system_info), indent=2)
                    except json.JSONDecodeError: system_info_text += system_info
                elif isinstance(system_info, dict): system_info_text += json.dumps(system_info, indent=2)
                else: system_info_text += str(system_info)
                user_messages.append({"type": "text", "text": system_info_text}) 
                
            user_messages.append({"type": "text", "text": "\n\nProvide your analysis strictly in the requested JSON format."}) 
            
            messages.append({
                "role": "user",
                "content": user_messages
            })
            
            # 4. Call LLM API
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=2000, do_sample=False)
                generation = generation[0][input_len:]

            response = self.processor.decode(generation, skip_special_tokens=True)
            print('Gemma3 output: ')
            print(response)
            
            raw_text = response
            # Attempt to parse the JSON payload (API should enforce this)
            if "json" in raw_text:
                print("Warning: Removing unexpected 'json' from LLM output.")
                raw_text = raw_text.replace("json", "")

            if "```" in raw_text:
                print("Warning: Removing Markdown code block backticks (``` ) from LLM output.")
                raw_text = raw_text.replace("```", "")

            result_json = json.loads(raw_text) 
            return result_json
            
        except FileNotFoundError:
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e:
             self.logger.error(f"Missing dependency: {e}")
             return None, {"error": "Missing dependency", "details": str(e)}
        except Exception as e:
            # Catch errors during image loading/preprocessing or FFT/NMF setup
            self.logger.exception(f"An unexpected error occurred before LLM call: {e}")
            return None, {"error": "An unexpected error occurred during analysis setup", "details": str(e)}
            
    def _analyze_image_base(self, image_path: str, system_info: dict | str | None, instruction_prompt: str) -> tuple[dict | None, dict | None]:
        """
        Internal helper method to handle common image analysis steps.
        """
        try:
            # 1. Load and Preprocess Image
            loaded_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(loaded_image)
            image = Image.fromarray(preprocessed_img_array)

            # 2. Optional FFT+NMF Analysis
            components_array, abundances_array = None, None
            fft_explanation = None # Store explanation if available
            if self.RUN_FFT_NMF:
                ws = None
                nc = None
                if self.FFT_NMF_AUTO_PARAMS:
                    # Note: LLM params estimation itself returns tuple (ws, nc, explanation)
                    auto_params = self._get_fft_nmf_params_from_llm(image, system_info)
                    if auto_params:
                         ws, nc, fft_explanation = auto_params # Unpack directly
                if fft_explanation:
                    self.logger.info(f"LLM Explanation for FFT/NMF params: {fft_explanation}")

                # Use config defaults if LLM fails or auto-params is off/failed
                if ws is None:
                    # Ensure preprocessed_img_array exists before accessing shape
                    default_ws = preprocessed_img_array.shape[0] // 16 if preprocessed_img_array is not None else 64
                    ws = self.fft_nmf_settings.get('window_size_x', default_ws) # Use correct key from config
                if nc is None:
                    nc = self.fft_nmf_settings.get('components', 4) # Use correct key from config

                # Calculate step size (ensure minimum of 1)
                step = max(1, ws // 4)
                # Run analysis with determined params
                components_array, abundances_array = self._run_fft_nmf_analysis(image_path, ws, nc, step)

            # 3. Construct Prompt
            
            messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction_prompt}]
            }]
            user_messages = []
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            user_messages.append({"type": "text", "text": analysis_request_text})
            user_messages.append({"type": "image", "image": image})

            # Add FFT/NMF results if available
            if components_array is not None and abundances_array is not None:
                user_messages.append({"type": "text", "text": "\n\nSupplemental Analysis Data (Sliding FFT + NMF Grayscale Images):"})
                num_components = min(components_array.shape[0], abundances_array.shape[0])
                img_format = 'JPEG' # Keep as JPEG for consistency/size
                self.logger.info(f"Adding {num_components} NMF components/abundances as PIL images to prompt.")

                for i in range(num_components):
                    try:
                        user_messages.append({"type": "text", "text": f"\nNMF Component {i+1} (Frequency Pattern - Grayscale):"})
                        user_messages.append({"type": "image", "image": normalize_and_convert_to_PIL(components_array[i], log_scale=True)})
                        user_messages.append({"type": "text", "text": f"\nNMF Abundance Map {i+1} (Spatial Distribution - Grayscale):"})
                        user_messages.append({"type": "image", "image": normalize_and_convert_to_PIL(abundances_array[i], log_scale=True)})
                    except Exception as convert_e:
                        self.logger.error(f"Failed to convert NMF result {i+1} to image bytes: {convert_e}")
                        user_messages.append({"type": "text", "text": f"\n(Error converting NMF result {i+1} image for prompt)"})
            else:
                user_messages.append({"type": "text", "text": "\n\n(No supplemental image analysis results are provided)"})

            if system_info:
                system_info_text = "\n\nAdditional System Information:\n"
                if isinstance(system_info, str):
                    try: system_info_text += json.dumps(json.loads(system_info), indent=2)
                    except json.JSONDecodeError: system_info_text += system_info
                elif isinstance(system_info, dict): system_info_text += json.dumps(system_info, indent=2)
                else: system_info_text += str(system_info)
                user_messages.append({"type": "text", "text": system_info_text}) 
                
            user_messages.append({"type": "text", "text": "\n\nProvide your analysis strictly in the requested JSON format."}) 
            
            messages.append({
                "role": "user",
                "content": user_messages
            })

            # 4. Call LLM API
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=2000, do_sample=False)
                generation = generation[0][input_len:]

            response = self.processor.decode(generation, skip_special_tokens=True)
            #print('Gemma2 output: ')
            #print(response)

            # 5. Parse and Validate Response
            result_json = None
            error_dict = None
            raw_text = None # Keep track of raw text for error reporting
            try:
                raw_text = response # Get raw text first
                # Attempt to extract JSON (more robustly)
                first_brace_index = raw_text.find('{')
                last_brace_index = raw_text.rfind('}')
                if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                    json_string = raw_text[first_brace_index : last_brace_index + 1]
                    result_json = json.loads(json_string)
                else:
                    raise ValueError("Could not find valid JSON object delimiters '{' and '}' in the response text.")

            except (json.JSONDecodeError, AttributeError, IndexError, ValueError) as e:
                error_details = str(e)
                error_raw_response = raw_text if raw_text is not None else getattr(response, 'text', 'N/A')
                self.logger.error(f"Error parsing Gemini JSON response: {e}")
                parsed_substring = json_string if 'json_string' in locals() else 'N/A'
                self.logger.debug(f"Attempted to parse substring: {parsed_substring[:500]}...")
                self.logger.debug(f"Original Raw response text: {error_raw_response[:500]}...")

            except Exception as e:
                 # Catch any other unexpected errors during API call or parsing
                 self.logger.exception(f"Unexpected error processing response: {e}")
                 error_dict = {"error": "Unexpected error processing LLM response", "details": str(e)}

            # Return either the successfully parsed JSON or the error dictionary
            return result_json, error_dict

        except FileNotFoundError:
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e:
             self.logger.error(f"Missing dependency: {e}")
             return None, {"error": "Missing dependency", "details": str(e)}
        except Exception as e:
            # Catch errors during image loading/preprocessing or FFT/NMF setup
            self.logger.exception(f"An unexpected error occurred before LLM call: {e}")
            return None, {"error": "An unexpected error occurred during analysis setup", "details": str(e)}

    def analyze_microscopy_image_for_structure_recommendations(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate DFT structure recommendations.
        """
        result_json, error_dict = self._analyze_image_base(
            image_path, system_info, MICROSCOPY_ANALYSIS_INSTRUCTIONS
        )

        if error_dict:
            return error_dict # Return error if base analysis failed

        if result_json is None:
             # Should not happen if error_dict is None, but safeguard
             return {"error": "Analysis failed unexpectedly after base processing."}

        # --- Specific Post-processing for Structure Recommendations ---
        detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided.")
        recommendations = result_json.get("structure_recommendations", [])
        valid_recommendations = []

        if not isinstance(recommendations, list):
            self.logger.warning(f"'structure_recommendations' was not a list: {recommendations}")
            recommendations = [] # Treat as empty list

        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in ["description", "scientific_interest", "priority"]):
                if isinstance(rec.get("priority"), int):
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Recommendation skipped due to invalid priority type: {rec}")
            else:
                 self.logger.warning(f"Recommendation skipped due to missing keys or incorrect format: {rec}")

        # Sort valid recommendations by priority
        sorted_recommendations = sorted(valid_recommendations, key=lambda x: x.get("priority", 99))

        if not sorted_recommendations and not detailed_analysis == "Analysis not provided.":
             self.logger.warning("Analysis successful but no valid recommendations found in the response.")
             # Optionally return a specific message, or just empty recommendations
             # return {"full_analysis": detailed_analysis, "recommendations": [], "warning": "No valid recommendations generated."}


        return {"full_analysis": detailed_analysis, "recommendations": sorted_recommendations}

    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims for literature comparison.
        """
        result_json, error_dict = self._analyze_image_base(
            image_path, system_info, MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict:
            return error_dict # Return error if base analysis failed

        if result_json is None:
             # Safeguard
             return {"error": "Analysis failed unexpectedly after base processing."}

        # --- Specific Post-processing for Scientific Claims ---
        detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided.")
        scientific_claims = result_json.get("scientific_claims", [])
        valid_claims = []

        if not isinstance(scientific_claims, list):
             self.logger.warning(f"'scientific_claims' was not a list: {scientific_claims}")
             scientific_claims = [] # Treat as empty

        for claim in scientific_claims:
            if isinstance(claim, dict) and all(k in claim for k in ["claim", "scientific_impact", "question", "keywords"]):
                # Optionally add validation for keywords being a list of strings etc.
                 valid_claims.append(claim)
            else:
                self.logger.warning(f"Claim skipped due to missing keys or incorrect format: {claim}")

        if not valid_claims and not detailed_analysis == "Analysis not provided.":
             self.logger.warning("Analysis successful but no valid claims found in the response.")
             # Optionally return a specific message or just empty claims
             # return {"full_analysis": detailed_analysis, "claims": [], "warning": "No valid claims generated."}

        return {"full_analysis": detailed_analysis, "claims": valid_claims}

    def _run_fft_nmf_analysis(self, image_path: str, window_size: int, n_components: int, window_step: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Runs the Sliding FFT + NMF analysis if enabled and available.
        Returns tuple (components_array, abundances_array) or (None, None).
        """
        try:
            self.logger.info("--- Starting Sliding FFT + NMF Analysis ---")
            fft_output_dir = self.fft_nmf_settings.get('output_dir', 'fft_nmf_results') # Use correct key
            os.makedirs(fft_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            fft_output_base = os.path.join(fft_output_dir, f"{safe_base_name}_analysis")

            analyzer = SlidingFFTNMF(
                window_size_x=window_size,
                window_size_y=window_size, # Assuming square windows based on config/LLM suggestions
                window_step_x=window_step,
                window_step_y=window_step, # Assuming square steps
                interpolation_factor=self.fft_nmf_settings.get('interpolation_factor', 2),
                zoom_factor=self.fft_nmf_settings.get('zoom_factor', 2),
                hamming_filter=self.fft_nmf_settings.get('hamming_filter', True),
                components=n_components
            )
            # analyze_image saves .npy files internally now
            components, abundances = analyzer.analyze_image(image_path, output_path=fft_output_base)

            self.logger.info(f"FFT+NMF analysis complete. Components shape: {components.shape}, Abundances shape: {abundances.shape}")
            # Abundances from analyze_image are already transposed to (n_components, h, w)
            return components, abundances

        except Exception as fft_e:
            self.logger.error(f"Sliding FFT + NMF analysis failed: {fft_e}", exc_info=True)

        return None, None # Return None if failed


    def _get_fft_nmf_params_from_llm(self, image, system_info) -> tuple[int | None, int | None, str | None]:
        """
        Makes a separate LLM call to estimate FFT/NMF parameters (window_size, n_components, explanation).
        Returns tuple (window_size, n_components, explanation) or (None, None, None) on failure.
        """
        self.logger.info("Attempting to get FFT/NMF parameters from LLM...")

        if system_info:
            # Simplified system info formatting for this internal call
             system_info_text = "\n\nAdditional System Information:\n" + str(system_info)[:1000] # Truncate for safety
        else:
            system_info_text = ""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": system_info_text + "\n\nOutput ONLY 'window_size', 'n_components', and 'explanation', do not include 'json' or backtick markdowns."}
                ]
            }
        ]
        
        
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
                generation = generation[0][input_len:]

            response = self.processor.decode(generation, skip_special_tokens=True)
            #print('Gemma output: ')
            #print(response)
            self.logger.debug(f"LLM parameter estimation raw response: {response}")

            raw_text = response
            # Attempt to parse the JSON payload (API should enforce this)
            if "json" in raw_text:
                print("Warning: Removing unexpected 'json' from LLM output.")
                raw_text = raw_text.replace("json", "")

            if "```" in raw_text:
                print("Warning: Removing Markdown code block backticks (``` ) from LLM output.")
                raw_text = raw_text.replace("```", "")

            result_json = json.loads(raw_text) 

            window_size = result_json.get("window_size")
            n_components = result_json.get("n_components")
            explanation = result_json.get("explanation", "No explanation provided.") # Default explanation

            # --- Validation ---
            params_valid = True
            if not isinstance(window_size, int) or window_size <= 0:
                self.logger.warning(f"LLM returned invalid window_size: {window_size} (Type: {type(window_size)})")
                params_valid = False
                window_size = None # Ensure invalid value isn't used

            if not isinstance(n_components, int) or not (1 <= n_components <= 16): # Allow 1, slightly wider range
                self.logger.warning(f"LLM returned invalid n_components: {n_components} (Type: {type(n_components)}). Expected int between 1-16.")
                params_valid = False
                n_components = None # Ensure invalid value isn't used

            if not isinstance(explanation, str) or not explanation.strip():
                 self.logger.warning(f"LLM returned invalid or empty explanation: {explanation}")
                 explanation = "Invalid or empty explanation received from LLM." # Provide default if bad


            if params_valid:
                self.logger.info(f"LLM successfully suggested parameters: window_size={window_size}, n_components={n_components}")
                return window_size, n_components, explanation
            else:
                self.logger.warning("LLM parameter suggestions failed validation. Falling back to defaults.")
                return None, None, explanation # Return explanation even if params bad

        except json.JSONDecodeError as json_e:
            self.logger.error(f"LLM parameter estimation response was not valid JSON: {json_e}. Raw text: '{getattr(response, 'text', 'N/A')[:500]}...'")
            return None, None, None
        except (AttributeError, IndexError, ValueError, TypeError) as parse_e:
             self.logger.error(f"Error parsing or validating LLM parameter response: {parse_e}", exc_info=True)
             return None, None, None
        except Exception as e:
            self.logger.error(f"LLM call for FFT/NMF parameters failed unexpectedly: {e}", exc_info=True)
            return None, None, None # Fallback on any error