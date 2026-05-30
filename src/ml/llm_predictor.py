"""
LLM-based startup success prediction via Hugging Face Transformers.
Handles prompting, caching, and probability parsing.
"""
import json
import os
import re
import torch
from typing import List, Dict, Optional
from pathlib import Path

# ============================================================================
# PROMPT TEMPLATES - Calibration-focused and Non-calibrated versions
# ============================================================================

# --- CALIBRATED PROMPTS (with base rate anchoring) ---

SYSTEM_PROMPT_CALIBRATED = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
You must provide well-calibrated probability estimates - when you say 70%, approximately 70% of such predictions should be correct.
Be conservative: most startups fail. Base rates matter.
- ~13% of startups raise follow-on funding within 2 years
- ~4% achieve a successful exit (acquisition or IPO) within 2 years"""

MOMENTUM_TEMPLATE_CALIBRATED = """Based on the following startup information, estimate the probability that this company will raise another funding round within the next 2 years.

**Company Profile:**
- Name: {name}
- Description: {description}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Current Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

Consider:
1. Market timing and industry trends
2. Funding trajectory (velocity, amounts)
3. Team size relative to funding stage
4. Product-market fit signals from description

Remember: the population base rate is ~13%, but adjust based on the company's specifics. Be calibrated.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.25"""

LIQUIDITY_TEMPLATE_CALIBRATED = """Based on the following startup information, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company Profile:**
- Name: {name}
- Description: {description}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Total Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

Consider:
1. Industry exit patterns (some sectors have more M&A activity)
2. Funding level vs typical exit thresholds
3. Company maturity (age, team size, funding rounds)
4. Market conditions for the sector

Remember: the population base rate is ~4%, but adjust based on the company's specifics. Be calibrated.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.10"""

# --- NON-CALIBRATED PROMPTS (no base rate anchoring) ---

SYSTEM_PROMPT_UNCALIBRATED = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Provide your best probability estimate based on the startup's characteristics."""

MOMENTUM_TEMPLATE_UNCALIBRATED = """Based on the following startup information, estimate the probability that this company will raise another funding round within the next 2 years.

**Company Profile:**
- Name: {name}
- Description: {description}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Current Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.35"""

LIQUIDITY_TEMPLATE_UNCALIBRATED = """Based on the following startup information, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company Profile:**
- Name: {name}
- Description: {description}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Total Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.15"""

# --- CHAIN-OF-THOUGHT PROMPTS (with reasoning before probability) ---

SYSTEM_PROMPT_COT = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Analyze startups carefully based on the provided data, explain your reasoning briefly, then give a well-calibrated probability estimate.
Base rates matter:
- ~13% of startups raise follow-on funding within 2 years
- ~4% achieve a successful exit (acquisition or IPO) within 2 years"""

MOMENTUM_TEMPLATE_COT = """Based on the following startup information, estimate the probability that this company will raise another funding round within the next 2 years.

**Company Profile:**
- Name: {name}
- Description: {description}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Current Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
1. Briefly explain your reasoning based on the data above (2-3 sentences)
2. Then give your final probability estimate

Base rate: ~13% of startups raise follow-on funding within 2 years.

End your response with:
PROBABILITY: [your estimate between 0.00 and 1.00]"""

LIQUIDITY_TEMPLATE_COT = """Based on the following startup information, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company Profile:**
- Name: {name}
- Description: {description}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Total Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
1. Briefly explain your reasoning based on the data above (2-3 sentences)
2. Then give your final probability estimate

Base rate: ~4% of startups achieve a successful exit within 2 years.

End your response with:
PROBABILITY: [your estimate between 0.00 and 1.00]"""

# --- DESCRIPTION-ONLY PROMPTS (minimal context, tests text value) ---

SYSTEM_PROMPT_DESCRIPTION_ONLY = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Based only on a startup's description, estimate the probability of success.
Be conservative - base rates matter:
- ~13% of startups raise follow-on funding within 2 years
- ~4% achieve a successful exit (acquisition or IPO) within 2 years"""

MOMENTUM_TEMPLATE_DESCRIPTION_ONLY = """Based on the following startup description, estimate the probability that this company will raise another funding round within the next 2 years.

**Startup Description:**
{description}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

Note: You only have the description to work with. Consider the market opportunity, product clarity, and business model implied by the description. The population base rate is ~13%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.25"""

LIQUIDITY_TEMPLATE_DESCRIPTION_ONLY = """Based on the following startup description, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Startup Description:**
{description}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

Note: You only have the description to work with. Consider the market opportunity, competitive positioning, and acquisition potential implied by the description. The population base rate is ~4%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.10"""

# --- FEATURES-ONLY PROMPTS (structured data only, no description text) ---

SYSTEM_PROMPT_FEATURES_ONLY = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Based only on structured data (funding, team, location), estimate the probability of success.
Be conservative: most startups fail. Base rates matter.
- ~13% of startups raise follow-on funding within 2 years
- ~4% achieve a successful exit (acquisition or IPO) within 2 years"""

MOMENTUM_TEMPLATE_FEATURES_ONLY = """Based on the following startup data, estimate the probability that this company will raise another funding round within the next 2 years.

**Company Data:**
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Current Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

Note: No company name or description is provided. Base your estimate purely on the structured data above.

Consider the funding trajectory, team size, and industry. The population base rate is ~13%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.25"""

LIQUIDITY_TEMPLATE_FEATURES_ONLY = """Based on the following startup data, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company Data:**
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Total Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

Note: No company name or description is provided. Base your estimate purely on the structured data above.

Consider the funding level, maturity, and market conditions. The population base rate is ~4%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.10"""

# --- NAME-ONLY PROMPTS (company name only, no features or description) ---

SYSTEM_PROMPT_NAME_ONLY = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Based only on the company name, estimate the probability of success.
If you recognize the company, use your knowledge. If not, provide your best guess."""

MOMENTUM_TEMPLATE_NAME_ONLY = """Based on the company name alone, estimate the probability that this startup will raise another funding round within the next 2 years.

**Company Name:** {name}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

If you recognize this company, use what you know to estimate. If you don't recognize it, provide your best guess based on the name alone.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.25"""

LIQUIDITY_TEMPLATE_NAME_ONLY = """Based on the company name alone, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company Name:** {name}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

If you recognize this company, use what you know to estimate. If you don't recognize it, provide your best guess based on the name alone.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.10"""

# --- NAME + FEATURES PROMPTS (name and structured data, no description) ---

SYSTEM_PROMPT_NAME_AND_FEATURES = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Based on the company name and structured data (funding, team, location), estimate the probability of success.
Be conservative: most startups fail. Base rates matter.
- ~13% of startups raise follow-on funding within 2 years
- ~4% achieve a successful exit (acquisition or IPO) within 2 years"""

MOMENTUM_TEMPLATE_NAME_AND_FEATURES = """Based on the following startup information, estimate the probability that this company will raise another funding round within the next 2 years.

**Company Profile:**
- Name: {name}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Current Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

Consider the company's funding trajectory, team size, and market. The population base rate is ~13%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.25"""

LIQUIDITY_TEMPLATE_NAME_AND_FEATURES = """Based on the following startup information, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company Profile:**
- Name: {name}
- Industry: {industries}
- Location: {city}
- Founded: {founded_on_year}
- Total Funding: ${total_funding_usd:,.0f}
- Funding Rounds: {num_funding_rounds}
- Employees: {employee_count}
- Number of Founders: {founder_count}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

Consider the company's funding level, maturity, and market conditions. The population base rate is ~4%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.10"""

# --- NAME + DESCRIPTION PROMPTS (name and description text, no structured data) ---

SYSTEM_PROMPT_NAME_AND_DESCRIPTION = """You are an expert venture capital analyst with 20 years of experience evaluating startups.
Based on the company name and description, estimate the probability of success.
Be conservative: most startups fail. Base rates matter.
- ~13% of startups raise follow-on funding within 2 years
- ~4% achieve a successful exit (acquisition or IPO) within 2 years"""

MOMENTUM_TEMPLATE_NAME_AND_DESCRIPTION = """Based on the following startup information, estimate the probability that this company will raise another funding round within the next 2 years.

**Company:**
- Name: {name}
- Description: {description}

**Your Task:**
Estimate the probability (0.00 to 1.00) that this startup will successfully raise another funding round within 2 years.

Consider what the description reveals about product-market fit and growth potential. The population base rate is ~13%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.25"""

LIQUIDITY_TEMPLATE_NAME_AND_DESCRIPTION = """Based on the following startup information, estimate the probability that this company will achieve a successful exit (acquisition or IPO) within the next 2 years.

**Company:**
- Name: {name}
- Description: {description}

**Your Task:**
Estimate the probability (0.00 to 1.00) of a successful exit (acquisition or IPO) within 2 years.

Consider what the description reveals about market position and exit potential. The population base rate is ~4%.

Respond with ONLY a decimal probability between 0.00 and 1.00, nothing else.
Example: 0.10"""


# ============================================================================
# HUGGING FACE CLIENT
# ============================================================================

class HuggingFaceClient:
    """Hugging Face Transformers client for text generation."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 32,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        token: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.token = token

        # Lazy loading
        self._model = None
        self._tokenizer = None
        self._pipeline = None

    def _load_model(self):
        """Load model and tokenizer on first use."""
        if self._pipeline is not None:
            return

        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"Loading model: {self.model_name}")

        # Determine torch dtype
        if self.torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Quantization config
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Determine device map
        if self.device == "auto":
            device_map = "auto"
        elif self.device == "cpu":
            device_map = "cpu"
            dtype = torch.float32  # CPU doesn't support float16 well
        else:
            device_map = self.device

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.token,
        )

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "token": self.token,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Create pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            torch_dtype=dtype,
            device_map=device_map,
        )

        print(f"Model loaded on device: {self._model.device}")

    def generate(self, prompt: str, system: str = None, temperature: float = 0.0) -> str:
        """Generate a single response."""
        self._load_model()

        # Format prompt based on model type
        if "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
            # Chat format for instruction-tuned models
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Simple format for base models
            if system:
                formatted_prompt = f"{system}\n\n{prompt}\n\nAnswer:"
            else:
                formatted_prompt = f"{prompt}\n\nAnswer:"

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "return_full_text": False,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        # Generate
        outputs = self._pipeline(formatted_prompt, **gen_kwargs)
        response = outputs[0]["generated_text"].strip()

        return response

    def is_available(self) -> bool:
        """Check if model can be loaded (basic check)."""
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            return True
        except Exception:
            return False


# ============================================================================
# PREDICTION CACHE
# ============================================================================

class PredictionCache:
    """Disk-based JSONL cache for LLM predictions.

    Each process writes to its own JSONL file (append-only, one JSON object
    per line) to avoid concurrent-write corruption.  On init, all existing
    JSONL files in the cache directory are loaded into memory so every agent
    benefits from predictions made by earlier / parallel agents.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Per-process output file keyed by SLURM job ID or PID
        run_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        self.cache_file = self.cache_dir / f"llm_predictions_{run_id}.jsonl"

        # Load all existing predictions from every JSONL file
        self.cache: dict[str, float] = {}
        self._load_all()

    def _load_all(self):
        """Load predictions from all JSONL files in the cache directory."""
        for path in sorted(self.cache_dir.glob("llm_predictions_*.jsonl")):
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry["prob"]
            except Exception as e:
                print(f"  Warning: failed to load cache file {path}: {e}")
        # Also load legacy JSON cache if it exists
        legacy = self.cache_dir / "llm_predictions.json"
        if legacy.exists() and legacy.stat().st_size > 0:
            try:
                with open(legacy, "r") as f:
                    data = json.load(f)
                self.cache.update(data)
                print(f"  Loaded {len(data)} predictions from legacy cache")
            except Exception:
                pass
        if self.cache:
            print(f"  Loaded {len(self.cache)} cached LLM predictions total")

    def _key(self, model: str, task: str, startup_id: str) -> str:
        model_key = model.replace("/", "_").replace(":", "_")
        return f"{model_key}:{task}:{startup_id}"

    def get(self, model: str, task: str, startup_id: str) -> Optional[float]:
        return self.cache.get(self._key(model, task, startup_id))

    def set(self, model: str, task: str, startup_id: str, prob: float):
        key = self._key(model, task, startup_id)
        self.cache[key] = prob
        # Append single line — atomic enough on local FS, no truncation risk
        with open(self.cache_file, "a") as f:
            f.write(json.dumps({"key": key, "prob": prob}) + "\n")
            f.flush()
            os.fsync(f.fileno())


# ============================================================================
# PROBABILITY PARSING
# ============================================================================

def parse_probability(response: str) -> float:
    """Extract probability from LLM response. Returns 0.5 on failure."""
    text = response.strip()
    text_lower = text.lower()

    # Pattern 0: "PROBABILITY: 0.XX" format (for chain-of-thought) - check first
    match = re.search(r"probability[:\s]+(\d*\.?\d+)", text_lower)
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val = val / 100
        return max(0.0, min(1.0, val))

    # Pattern 1: Plain decimal (0.75, .75)
    match = re.search(r"(?:^|[^\d])(\d*\.?\d+)(?:[^\d]|$)", text_lower)
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val = val / 100  # Convert percentage
        return max(0.0, min(1.0, val))

    # Pattern 2: Percentage (75%, 75 percent)
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text_lower)
    if match:
        return max(0.0, min(1.0, float(match.group(1)) / 100))

    return 0.5  # Default fallback


# ============================================================================
# PROMPT BUILDER
# ============================================================================

class PromptBuilder:
    """Builds prompts from startup features."""

    def __init__(self, task: str, use_calibration: bool = True, use_chain_of_thought: bool = False, prompt_features: str = "full"):
        self.task = task
        self.use_calibration = use_calibration
        self.use_chain_of_thought = use_chain_of_thought
        self.prompt_features = prompt_features

        # Select template based on settings
        # Priority: ablation modes > chain_of_thought > calibration
        _templates = {
            "features_only":          (MOMENTUM_TEMPLATE_FEATURES_ONLY,          LIQUIDITY_TEMPLATE_FEATURES_ONLY),
            "description_only":       (MOMENTUM_TEMPLATE_DESCRIPTION_ONLY,       LIQUIDITY_TEMPLATE_DESCRIPTION_ONLY),
            "name_only":              (MOMENTUM_TEMPLATE_NAME_ONLY,              LIQUIDITY_TEMPLATE_NAME_ONLY),
            "name_and_features":      (MOMENTUM_TEMPLATE_NAME_AND_FEATURES,      LIQUIDITY_TEMPLATE_NAME_AND_FEATURES),
            "name_and_description":   (MOMENTUM_TEMPLATE_NAME_AND_DESCRIPTION,   LIQUIDITY_TEMPLATE_NAME_AND_DESCRIPTION),
        }
        if prompt_features in _templates:
            mom, liq = _templates[prompt_features]
            self.template = mom if task == "momentum" else liq
        elif use_chain_of_thought:
            self.template = MOMENTUM_TEMPLATE_COT if task == "momentum" else LIQUIDITY_TEMPLATE_COT
        elif use_calibration:
            self.template = MOMENTUM_TEMPLATE_CALIBRATED if task == "momentum" else LIQUIDITY_TEMPLATE_CALIBRATED
        else:
            self.template = MOMENTUM_TEMPLATE_UNCALIBRATED if task == "momentum" else LIQUIDITY_TEMPLATE_UNCALIBRATED

    def build(self, features: Dict) -> str:
        """Build prompt from feature dict, handling missing values.

        The raw_df column names may differ from the template placeholders
        (e.g. CSV has ``short_description`` but templates use ``{description}``).
        We check alternative column names before falling back to defaults.
        """
        import pandas as pd

        # Map template placeholder -> list of alternative column names in raw_df
        _aliases = {
            "description": ["short_description", "description"],
            "industries": ["category_list", "category_groups_list", "industries"],
            "city": ["city", "location_city"],
            "founded_on_year": ["founded_on_year", "founded_on"],
            "founder_count": ["founder_count", "num_founders"],
        }

        defaults = {
            "name": "Unknown",
            "description": "No description available",
            "industries": "Unknown",
            "city": "Unknown",
            "founded_on_year": "Unknown",
            "total_funding_usd": 0,
            "num_funding_rounds": 0,
            "employee_count": "Unknown",
            "founder_count": 0,
        }

        safe = {}
        for key, default in defaults.items():
            val = features.get(key)
            # Try alternative column names if primary key is missing
            if val is None and key in _aliases:
                for alias in _aliases[key]:
                    val = features.get(alias)
                    if val is not None:
                        break
            if val is None or (isinstance(val, float) and pd.isna(val)):
                safe[key] = default
            else:
                safe[key] = val

        return self.template.format(**safe)


# ============================================================================
# MAIN PREDICTOR CLASS
# ============================================================================

class LLMPredictor:
    """
    Predicts startup success using LLM.
    Supports momentum (next funding) and liquidity (exit) tasks.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir: str = "outputs/llm_cache",
        temperature: float = 0.0,
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        token: str = None,
        use_calibration: bool = True,
        use_chain_of_thought: bool = False,
        prompt_features: str = "full",  # "full", "description_only", "features_only", "name_only", "name_and_features", "name_and_description"
    ):
        # Chain-of-thought needs more tokens for reasoning
        max_new_tokens = 256 if use_chain_of_thought else 32

        self.client = HuggingFaceClient(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            token=token,
        )
        self.cache = PredictionCache(cache_dir)
        self.model_name = model_name
        self.temperature = temperature
        self.use_calibration = use_calibration
        self.use_chain_of_thought = use_chain_of_thought
        self.prompt_features = prompt_features

        # Cache key includes all settings to separate cached predictions
        calib_suffix = "_calib" if use_calibration else "_uncalib"
        cot_suffix = "_cot" if use_chain_of_thought else ""
        features_suffix = {
            "description_only": "_desconly",
            "features_only": "_featonly",
            "name_only": "_nameonly",
            "name_and_features": "_namefeat",
            "name_and_description": "_namedesc",
        }.get(prompt_features, "")
        self.cache_model_key = f"{model_name}{calib_suffix}{cot_suffix}{features_suffix}"

        # Select system prompt based on settings
        _system_prompts = {
            "features_only": SYSTEM_PROMPT_FEATURES_ONLY,
            "description_only": SYSTEM_PROMPT_DESCRIPTION_ONLY,
            "name_only": SYSTEM_PROMPT_NAME_ONLY,
            "name_and_features": SYSTEM_PROMPT_NAME_AND_FEATURES,
            "name_and_description": SYSTEM_PROMPT_NAME_AND_DESCRIPTION,
        }
        if prompt_features in _system_prompts:
            self.system_prompt = _system_prompts[prompt_features]
        elif use_chain_of_thought:
            self.system_prompt = SYSTEM_PROMPT_COT
        elif use_calibration:
            self.system_prompt = SYSTEM_PROMPT_CALIBRATED
        else:
            self.system_prompt = SYSTEM_PROMPT_UNCALIBRATED

        # Prompt builders for each task
        self.prompt_builders = {
            "momentum": PromptBuilder("momentum", use_calibration=use_calibration, use_chain_of_thought=use_chain_of_thought, prompt_features=prompt_features),
            "liquidity": PromptBuilder("liquidity", use_calibration=use_calibration, use_chain_of_thought=use_chain_of_thought, prompt_features=prompt_features),
        }

        calib_status = "enabled" if use_calibration else "disabled"
        cot_status = "enabled" if use_chain_of_thought else "disabled"
        features_status = prompt_features
        print(f"  LLM Predictor: Features={features_status}, Calibration={calib_status}, CoT={cot_status}")

    def predict_single(self, features: Dict, task: str, use_cache: bool = True) -> float:
        """Predict probability for a single startup."""
        startup_id = str(features.get("org_uuid", features.get("name", "unknown")))

        # Check cache (uses cache_model_key which includes calibration status)
        if use_cache:
            cached = self.cache.get(self.cache_model_key, task, startup_id)
            if cached is not None:
                return cached

        # Build prompt and generate
        prompt = self.prompt_builders[task].build(features)
        response = self.client.generate(prompt, system=self.system_prompt, temperature=self.temperature)
        prob = parse_probability(response)

        # Cache result (uses cache_model_key which includes calibration status)
        if use_cache:
            self.cache.set(self.cache_model_key, task, startup_id, prob)

        return prob

    def predict_batch(
        self,
        feature_dicts: List[Dict],
        task: str,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[float]:
        """Predict probabilities for a batch of startups."""
        results = []

        for i, features in enumerate(feature_dicts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  LLM predictions: {i + 1}/{len(feature_dicts)}")
            results.append(self.predict_single(features, task, use_cache))

        return results
