import asyncio
import json
import re
import requests
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Change this if your MCP server is somewhere else.
SERVER_RELATIVE_PATH = Path("Servers") / "filesystem_server.py"

# Maximum number of tool/model iterations for one workflow phase.
MAX_PHASE_ITERATIONS = 8


# ============================================================
# CONVERSATION
# ============================================================

ConversationWindow = []


def add_user_message(message):
    ConversationWindow.append({
        "role": "user",
        "content": message
    })


def add_ai_message(message):
    ConversationWindow.append({
        "role": "assistant",
        "content": message
    })


def add_assistant_tool_calls(content, tool_calls):
    ConversationWindow.append({
        "role": "assistant",
        "content": content or "",
        "tool_calls": tool_calls
    })


def add_tool_message(tool_id, tool_name, result):
    ConversationWindow.append({
        "role": "tool",
        "tool_call_id": tool_id,
        "name": tool_name,
        "content": json.dumps(result, ensure_ascii=False)
    })


# ============================================================
# AUTHORITATIVE WORKFLOW STATE
#
# IMPORTANT:
# Qwen does NOT control this state.
# Python controls it.
# ============================================================

workflow_state = {
    "phase": "WAITING_FOR_DOCUMENT",

    "document": {
        "loaded": False,
        "path": None,
        "content": None
    },

    "resume_facts": None,

    "learner_input": {
        "target_role": None,
        "career_goal": None,
        "skill_ratings": {},
        "learning_preferences": {}
    },

    "learner_profile": None,

    "skill_gaps": None,

    "resources": [],

    "roadmap": None,

    "progress_initialized": False,

    "completed": False
}


# ============================================================
# EDUPATH MCP TOOLS ONLY
#
# Do NOT expose generic filesystem tools to Qwen.
# ============================================================

ALLOWED_TOOLS = {
    "read_document",
    "save_learner_profile",
    "save_skill_gap_analysis",
    "get_learner_profile",
    "get_skill_gap_analysis",
    "save_learning_roadmap",
    "get_learning_roadmap",
    "update_learning_progress",
    "get_learning_progress",
    "search_learning_resources",
    "read_web_page",
}


# ============================================================
# TOOL CONVERSION
# ============================================================

def convert_mcp_tool_to_ollama(tool):

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }
    }


# ============================================================
# MCP RESULT EXTRACTION
# ============================================================

def extract_mcp_result(mcp_result):

    # Structured result
    if getattr(mcp_result, "structured_content", None) is not None:
        return mcp_result.structured_content

    # Normal MCP text content
    if getattr(mcp_result, "content", None):

        text_parts = []

        for item in mcp_result.content:

            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)

        if text_parts:

            combined_text = "\n".join(text_parts)

            try:
                return json.loads(combined_text)

            except json.JSONDecodeError:
                return combined_text

    return {
        "error": "MCP tool returned no usable result"
    }


# ============================================================
# MCP ERROR DETECTION
# ============================================================

def tool_result_failed(mcp_result, extracted_result):

    # MCP explicitly reports an error
    if getattr(mcp_result, "is_error", False):
        return True

    # Some of your tools return:
    #
    # {
    #     "error": "..."
    # }
    #
    # without setting is_error=True.

    if isinstance(extracted_result, dict):

        if "error" in extracted_result:
            return True

    return False


# ============================================================
# OLLAMA
# ============================================================

def prompt_ai(payload):

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

    except requests.RequestException as exc:

        print("\nOllama connection error:")
        print(exc)

        return None

    if response.status_code != 200:

        print("\nOllama HTTP error:")
        print(response.status_code)
        print(response.text)

        return None

    return response


# ============================================================
# JSON EXTRACTION
#
# Qwen sometimes wraps JSON in markdown.
# ============================================================

def extract_json_from_text(text):

    if not text:
        return None

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ```json ... ```
    match = re.search(
        r"```json\s*(.*?)\s*```",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# PROFILE VALIDATION
# ============================================================

def validate_resume_facts(data):

    if not isinstance(data, dict):
        return False

    allowed = {
        "name",
        "skills",
        "experience",
        "projects"
    }

    # Remove anything outside our allowed structure.
    data = {
        key: value
        for key, value in data.items()
        if key in allowed
    }

    if "name" not in data:
        data["name"] = None

    if not isinstance(data.get("skills", []), list):
        data["skills"] = []

    if not isinstance(data.get("experience", []), list):
        data["experience"] = []

    if not isinstance(data.get("projects", []), list):
        data["projects"] = []

    return data


# ============================================================
# BUILD COMPACT LEARNER PROFILE
#
# IMPORTANT:
# Qwen does NOT construct this.
#
# Python constructs it from:
#
# 1. Resume facts
# 2. Explicit user answers
# ============================================================

def build_learner_profile():

    facts = workflow_state["resume_facts"]

    learner_input = workflow_state["learner_input"]

    profile = {

        "learner": {
            "name": facts.get("name")
        },

        "current_skills": facts.get("skills", []),

        "experience": facts.get("experience", []),

        "projects": facts.get("projects", []),

        "target_role": learner_input["target_role"],

        "career_goal": learner_input["career_goal"],

        "skill_ratings": learner_input["skill_ratings"],

        "learning_preferences": learner_input["learning_preferences"]
    }

    return profile


# ============================================================
# PROFILE VALIDATION
# ============================================================

def validate_profile(profile):

    required_fields = {
        "learner",
        "current_skills",
        "experience",
        "projects",
        "target_role",
        "career_goal",
        "skill_ratings",
        "learning_preferences"
    }

    if not required_fields.issubset(profile.keys()):
        return False

    if not profile["target_role"]:
        return False

    if not profile["career_goal"]:
        return False

    if not isinstance(profile["current_skills"], list):
        return False

    if not isinstance(profile["experience"], list):
        return False

    if not isinstance(profile["projects"], list):
        return False

    if not isinstance(profile["skill_ratings"], dict):
        return False

    return True


# ============================================================
# CURRENT WORKFLOW CONTEXT
# ============================================================

def workflow_context():

    state = workflow_state

    return f"""
CURRENT EDUPATH STATE

phase: {state["phase"]}

document_loaded: {state["document"]["loaded"]}
document_path: {state["document"]["path"]}

resume_facts_extracted:
{json.dumps(state["resume_facts"], indent=2, ensure_ascii=False)
 if state["resume_facts"] else "NOT YET"}

learner_profile:
{json.dumps(state["learner_profile"], indent=2, ensure_ascii=False)
 if state["learner_profile"] else "NOT YET"}

skill_gaps:
{json.dumps(state["skill_gaps"], indent=2, ensure_ascii=False)
 if state["skill_gaps"] else "NOT YET"}

resources:
{json.dumps(state["resources"], indent=2, ensure_ascii=False)}

roadmap:
{json.dumps(state["roadmap"], indent=2, ensure_ascii=False)
 if state["roadmap"] else "NOT YET"}

RULE:

Python controls the workflow state.

Do NOT assume that a step has been completed unless the
Python state says it has been completed.

Do NOT invent learner facts.
"""


# ============================================================
# PAYLOAD
# ============================================================

def build_payload(system_prompt, tools):

    messages = []

    messages.append({
        "role": "system",
        "content": (
            system_prompt
            + "\n\n"
            + workflow_context()
        )
    })

    messages.extend(ConversationWindow)

    return {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False
    }


# ============================================================
# PHASE PROMPTS
# ============================================================

DOCUMENT_PROMPT = """
You are EduPath.

CURRENT PHASE:
DOCUMENT

The learner has provided a document.

Your ONLY job in this phase is:

1. Call read_document using the exact document path supplied
   by the learner.
2. Do not call any other tool.
3. Do not create a learner profile.
4. Do not invent information.
5. Do not ask the learner for skills that can be extracted
   from the document.

After read_document returns, stop.
"""


RESUME_EXTRACTION_PROMPT = """
You are EduPath.

CURRENT PHASE:
RESUME_EXTRACTION

The document has already been read.

The complete document content is available in the previous
tool result.

Extract ONLY facts explicitly supported by the document.

Return ONLY valid JSON.

Required format:

{
    "name": "string or null",

    "skills": [
        {
            "name": "Java",
            "evidence": "resume"
        }
    ],

    "experience": [
        {
            "company": "string",
            "role": "string",
            "period": "string",
            "responsibilities": []
        }
    ],

    "projects": [
        {
            "name": "string",
            "technologies": [],
            "description": "string"
        }
    ]
}

IMPORTANT:

A skill listed in the resume is evidence that the resume
claims the learner has that skill.

Do NOT convert it into a numeric proficiency.

If the resume says:

Python (Basic)

preserve:

{
    "name": "Python",
    "evidence": "resume states Basic"
}

Do not invent:

{
    "rating": 2
}

Do not invent target roles.

Do not invent career goals.

Do not invent experience years.

Do not invent projects.

Do not invent technologies.

Only extract what the document supports.
"""


PROFILE_QUESTION_PROMPT = """
You are EduPath.

CURRENT PHASE:
PROFILE_INFORMATION

The resume facts have already been extracted.

The following information is still required from the learner:

- target role
- career goal
- optionally skill ratings for important skills

Do NOT ask for information that already exists in the resume.

Ask the learner for the missing information.

Use exactly:

<NEEDS_USER_INPUT>
your question
</NEEDS_USER_INPUT>

Do not create the profile yet.
"""


SKILL_GAP_PROMPT = """
You are EduPath.

CURRENT PHASE:
SKILL_GAP_ANALYSIS

Use ONLY the authoritative learner profile provided in the
workflow state.

Compare:

current skills
+
experience
+
projects

against:

target role
+
career goal

Return ONLY valid JSON.

Format:

{
    "target_role": "...",

    "gaps": [
        {
            "skill": "...",
            "current_evidence": "...",
            "required_for_role": "...",
            "gap": "...",
            "priority": "high|medium|low"
        }
    ]
}

Rules:

Do not invent current skills.

Do not invent experience.

Do not invent skill ratings.

Do not claim that the learner knows something that is not
supported by the profile.

The skill gap is an AI-derived analysis, not a learner fact.
"""


RESOURCE_PROMPT = """
You are EduPath.

CURRENT PHASE:
RESOURCE_RESEARCH

Use the saved skill-gap analysis.

For important gaps, use search_learning_resources.

Rules:

1. Search only for actual identified gaps.
2. Do not search for skills the learner already knows unless
   the gap analysis explicitly says advanced knowledge is needed.
3. Do not invent search results.
4. If a search fails, do not repeatedly call the same search.
5. A failed search does not mean the learner has a new gap.
"""


ROADMAP_PROMPT = """
You are EduPath's roadmap generation engine.

Your ONLY task is to convert the supplied learner profile, skill-gap analysis,
and researched resources into a 12-week learning roadmap.

Target role: Senior Software Engineer
Career goal: Software Engineer who is an expert in Agentic AI and Spring Boot.

OUTPUT RULES:
1. Return ONLY one valid JSON object.
2. The first character MUST be { and the last character MUST be }.
3. No Markdown, code fences, explanations, headings, summaries, or text outside JSON.
4. Do NOT perform web searches.
5. Do NOT invent URLs or resources. If a week has no suitable supplied resource, use an empty resources array.
6. Use only supplied resources when assigning resources to weeks.
7. Create exactly 12 weeks, numbered 1 through 12.
8. Every week must have status "not_started" and progressPercentage 0.
9. Every practice task must have status "not_started".

Required structure:

{
  "roadmap": {
    "targetRole": "Senior Software Engineer",
    "careerGoal": "Software Engineer who is an expert in Agentic AI and Spring Boot",
    "timelineWeeks": 12,
    "currentSkills": [],
    "priorityGaps": [],
    "weeks": [
      {
        "weekNumber": 1,
        "title": "string",
        "status": "not_started",
        "progressPercentage": 0,
        "objectives": [],
        "resources": [
          {
            "title": "string",
            "url": "string",
            "type": "string"
          }
        ],
        "practiceTasks": [
          {
            "id": "w1-t1",
            "title": "string",
            "status": "not_started"
          }
        ],
        "completionCriteria": []
      }
    ]
  }
}

The roadmap must progress logically from Agentic AI and Spring Boot foundations
through implementation, tool calling, MCP, RAG, architecture, production
engineering, cloud, and a capstone.

Use the learner profile and supplied skill gaps as authoritative inputs.
Do not invent learner experience.

OUTPUT JSON ONLY. NO OTHER TEXT.
"""


# ============================================================
# USER INPUT EXTRACTION
# ============================================================

NEEDS_USER_INPUT_PATTERN = re.compile(
    r"<NEEDS_USER_INPUT>\s*(.*?)\s*</NEEDS_USER_INPUT>",
    re.IGNORECASE | re.DOTALL
)


def extract_user_question(content):

    match = NEEDS_USER_INPUT_PATTERN.search(content or "")

    if not match:
        return None

    return match.group(1).strip()


# ============================================================
# DOCUMENT PATH EXTRACTION
# ============================================================

def extract_document_path(user_message):

    # Windows quoted path
    match = re.search(
        r'"([A-Za-z]:\\[^"]+\.(?:pdf|txt))"',
        user_message,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    # Windows unquoted path
    match = re.search(
        r'([A-Za-z]:\\[^\s"]+\.(?:pdf|txt))',
        user_message,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# MCP TOOL ARGUMENT NORMALIZATION
# ============================================================

def normalize_arguments(arguments):

    if isinstance(arguments, str):

        try:
            arguments = json.loads(arguments)

        except json.JSONDecodeError:

            return None

    if not isinstance(arguments, dict):
        return None

    return arguments


# ============================================================
# TOOL PHASE PERMISSION
# ============================================================

def tool_allowed_in_phase(tool_name):

    phase = workflow_state["phase"]

    allowed = {

        "WAITING_FOR_DOCUMENT": {
            "read_document"
        },

        "DOCUMENT": {
            "read_document"
        },

        "RESUME_EXTRACTION": set(),

        "PROFILE_INFORMATION": set(),

        "SAVE_PROFILE": {
            "save_learner_profile"
        },

        "SKILL_GAP_ANALYSIS": {
            "get_learner_profile",
            "save_skill_gap_analysis"
        },

        "RESOURCE_RESEARCH": {
            "search_learning_resources",
            "read_web_page"
        },

        "ROADMAP": {
            "save_learning_roadmap"
        },

        "PROGRESS": {
            "update_learning_progress"
        },

        "COMPLETE": {
            "get_learning_progress"
        }
    }

    return tool_name in allowed.get(phase, set())


# ============================================================
# EXECUTE ONE QWEN TURN
# ============================================================

async def run_qwen_turn(
    session,
    ollama_tools,
    system_prompt
):

    payload = build_payload(
        system_prompt,
        ollama_tools
    )

    response = prompt_ai(payload)

    if response is None:
        return None

    result = response.json()

    message = result.get("message", {})

    return message


# ============================================================
# DOCUMENT PHASE
# ============================================================

async def process_document_phase(
    session,
    ollama_tools
):

    workflow_state["phase"] = "DOCUMENT"

    for iteration in range(MAX_PHASE_ITERATIONS):

        message = await run_qwen_turn(
            session,
            ollama_tools,
            DOCUMENT_PROMPT
        )

        if message is None:
            return False

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            print("\nQwen did not call read_document.")
            return False

        add_assistant_tool_calls(
            message.get("content", ""),
            tool_calls
        )

        for tool in tool_calls:

            tool_name = tool["function"]["name"]

            arguments = normalize_arguments(
                tool["function"]["arguments"]
            )

            if tool_name != "read_document":

                result = {
                    "error": (
                        "Only read_document is allowed during "
                        "the document phase."
                    )
                }

                add_tool_message(
                    tool["id"],
                    tool_name,
                    result
                )

                continue

            if arguments is None:

                result = {
                    "error": "Invalid tool arguments."
                }

                add_tool_message(
                    tool["id"],
                    tool_name,
                    result
                )

                continue

            # ------------------------------------------------
            # IMPORTANT:
            #
            # We DO NOT block rereading by returning an error.
            #
            # The phase controller prevents unnecessary calls.
            # ------------------------------------------------

            mcp_result = await session.call_tool(
                tool_name,
                arguments=arguments
            )

            extracted = extract_mcp_result(mcp_result)

            failed = tool_result_failed(
                mcp_result,
                extracted
            )

            print("\nMCP read_document result:")

            if isinstance(extracted, str):
                print(extracted[:3000])

            else:
                print(
                    json.dumps(
                        extracted,
                        indent=2,
                        ensure_ascii=False
                    )[:3000]
                )

            add_tool_message(
                tool["id"],
                tool_name,
                extracted
            )

            if failed:

                print("\nDocument reading failed.")

                return False

            # ----------------------------------------------
            # SAVE THE ACTUAL DOCUMENT CONTENT IN PYTHON
            # ----------------------------------------------

            document_content = None

            if isinstance(extracted, dict):

                document_content = (
                    extracted.get("text")
                    or extracted.get("content")
                    or extracted.get("document")
                )

            elif isinstance(extracted, str):

                document_content = extracted

            if not document_content:

                print(
                    "\nread_document succeeded but returned "
                    "no document text."
                )

                return False

            workflow_state["document"]["loaded"] = True

            workflow_state["document"]["content"] = (
                document_content
            )

            workflow_state["document"]["path"] = (
                arguments.get("file_path")
                or arguments.get("path")
            )

            return True

    return False


# ============================================================
# RESUME EXTRACTION
# ============================================================

async def extract_resume_facts():

    workflow_state["phase"] = "RESUME_EXTRACTION"

    document_content = workflow_state["document"]["content"]

    # We intentionally use a fresh mini conversation here.
    #
    # This prevents old tool-call history from confusing Qwen.

    messages = [

        {
            "role": "system",
            "content": RESUME_EXTRACTION_PROMPT
        },

        {
            "role": "user",
            "content": (
                "Extract facts from this document:\n\n"
                + document_content
            )
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

    response = prompt_ai(payload)

    if response is None:
        return False

    result = response.json()

    content = result.get(
        "message",
        {}
    ).get(
        "content",
        ""
    )

    print("\nResume extraction:")
    print(content)

    facts = extract_json_from_text(content)

    if facts is None:

        print(
            "\nQwen did not return valid resume JSON."
        )

        return False

    facts = validate_resume_facts(facts)

    workflow_state["resume_facts"] = facts

    return True


# ============================================================
# GET USER INFORMATION
# ============================================================

def request_missing_profile_information():

    missing = []

    if not workflow_state["learner_input"]["target_role"]:
        missing.append("target role")

    if not workflow_state["learner_input"]["career_goal"]:
        missing.append("career goal")

    print("\n========================================")
    print("Additional learner information required")
    print("========================================")

    if "target role" in missing:

        answer = input(
            "\nWhat role are you preparing for? "
        ).strip()

        if answer:

            workflow_state[
                "learner_input"
            ]["target_role"] = answer

    if "career goal" in missing:

        answer = input(
            "\nWhat is your career goal? "
        ).strip()

        if answer:

            workflow_state[
                "learner_input"
            ]["career_goal"] = answer

    return (
        bool(workflow_state["learner_input"]["target_role"])
        and
        bool(workflow_state["learner_input"]["career_goal"])
    )


# ============================================================
# SAVE PROFILE
# ============================================================

async def save_profile(
    session,
    mcp_tools
):

    workflow_state["phase"] = "SAVE_PROFILE"

    profile = build_learner_profile()

    if not validate_profile(profile):

        print(
            "\nProfile validation failed."
        )

        return False

    workflow_state["learner_profile"] = profile

    profile_json = json.dumps(
        profile,
        indent=2,
        ensure_ascii=False
    )

    print("\n========================================")
    print("Learner Profile")
    print("========================================")

    print(profile_json)

    mcp_result = await session.call_tool(
        "save_learner_profile",
        arguments={
            "profile_json": profile_json
        }
    )

    extracted = extract_mcp_result(mcp_result)

    if tool_result_failed(
        mcp_result,
        extracted
    ):

        print("\nFailed to save learner profile:")
        print(extracted)

        return False

    print("\nLearner profile saved successfully.")

    return True


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

async def create_skill_gap_analysis(
    session,
    ollama_tools
):

    workflow_state["phase"] = "SKILL_GAP_ANALYSIS"

    messages = [

        {
            "role": "system",
            "content": SKILL_GAP_PROMPT
        },

        {
            "role": "user",
            "content": (
                "Analyze this authoritative learner profile:\n\n"
                +
                json.dumps(
                    workflow_state["learner_profile"],
                    indent=2,
                    ensure_ascii=False
                )
            )
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

    response = prompt_ai(payload)

    if response is None:
        return False

    result = response.json()

    content = result.get(
        "message",
        {}
    ).get(
        "content",
        ""
    )

    print("\nSkill gap analysis:")
    print(content)

    gaps = extract_json_from_text(content)

    if gaps is None:

        print(
            "\nInvalid skill gap JSON."
        )

        return False

    workflow_state["skill_gaps"] = gaps

    gap_json = json.dumps(
        gaps,
        indent=2,
        ensure_ascii=False
    )

    mcp_result = await session.call_tool(
        "save_skill_gap_analysis",
        arguments={
            "gap_analysis_json": gap_json
        }
    )

    extracted = extract_mcp_result(mcp_result)

    if tool_result_failed(
        mcp_result,
        extracted
    ):

        print("\nFailed to save skill gaps:")
        print(extracted)

        return False

    print(
        "\nSkill-gap analysis saved successfully."
    )

    return True


# ============================================================
# RESOURCE RESEARCH
# ============================================================

async def research_resources(
    session,
    ollama_tools
):

    workflow_state["phase"] = "RESOURCE_RESEARCH"

    gaps = workflow_state["skill_gaps"]

    if not isinstance(gaps, dict):
        return False

    gap_list = gaps.get(
        "gaps",
        []
    )

    # Only research the most important gaps.
    gap_list = gap_list[:5]

    for gap in gap_list:

        topic = gap.get("skill")

        if not topic:
            continue

        priority = gap.get(
            "priority",
            "medium"
        )

        if priority == "low":
            continue

        # Determine a reasonable search level.
        level = "beginner"

        search_args = {
            "topic": topic,
            "level": level
        }

        print(
            f"\nSearching resources for: {topic}"
        )

        mcp_result = await session.call_tool(
            "search_learning_resources",
            arguments=search_args
        )

        extracted = extract_mcp_result(
            mcp_result
        )

        if tool_result_failed(
            mcp_result,
            extracted
        ):

            print(
                f"Resource search failed for {topic}"
            )

            # IMPORTANT:
            #
            # Do NOT ask Qwen to endlessly retry.
            #
            continue

        if isinstance(extracted, dict):

            if "resources" in extracted:

                workflow_state[
                    "resources"
                ].extend(
                    extracted["resources"]
                )

            elif "error" not in extracted:

                workflow_state[
                    "resources"
                ].append(extracted)

        elif isinstance(extracted, list):

            workflow_state[
                "resources"
            ].extend(extracted)

    return True


# ============================================================
# STRUCTURED JSON GENERATION
# ============================================================

async def generate_json_with_repair(
    purpose,
    context,
    system_prompt,
    max_attempts=2
):
    """
    Ask the local Ollama model for JSON and perform one bounded repair
    attempt if the model returns prose, markdown, or malformed JSON.

    Python remains authoritative: this helper only generates content.
    """

    base_user_prompt = (
        f"Generate the requested {purpose} from the supplied data.\n\n"
        "AUTHORITATIVE DATA:\n"
        + json.dumps(
            context,
            indent=2,
            ensure_ascii=False
        )
        + "\n\n"
        "OUTPUT REQUIREMENT:\n"
        "Return ONLY a valid JSON object. "
        "Do not include Markdown, code fences, explanations, headings, "
        "or any text before or after the JSON."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": base_user_prompt
        }
    ]

    for attempt in range(1, max_attempts + 1):

        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "format": "json"
        }

        response = prompt_ai(payload)

        if response is None:
            return None

        try:
            result = response.json()
        except Exception:
            result = {}

        message = result.get("message") or {}

        content = message.get("content") or ""

        parsed = extract_json_from_text(content)

        if isinstance(parsed, dict):
            return parsed

        if attempt < max_attempts:

            messages = [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n\nABSOLUTE RULE: "
                        "Your entire response must be one JSON object."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Regenerate it from the same authoritative data. "
                        "Return ONLY the JSON object.\n\n"
                        + base_user_prompt
                    )
                }
            ]

    return None


# ============================================================
# CREATE ROADMAP
# ============================================================

async def create_roadmap(session):

    workflow_state["phase"] = "ROADMAP"

    context = {
        "learner_profile": workflow_state["learner_profile"],
        "skill_gaps": workflow_state["skill_gaps"],
        "resources": workflow_state["resources"]
    }

    print("\nGenerating 12-week learning roadmap...")

    roadmap_response = await generate_json_with_repair(
        purpose="12-week learning roadmap",
        context=context,
        system_prompt=ROADMAP_PROMPT,
        max_attempts=2
    )

    if not isinstance(roadmap_response, dict):
        print(
            "\nThe AI did not return a valid roadmap JSON object."
        )
        return False

    roadmap_data = roadmap_response.get("roadmap")

    if not isinstance(roadmap_data, dict):
        print("\nRoadmap JSON is missing the 'roadmap' object.")
        return False

    weeks = roadmap_data.get("weeks")

    if not isinstance(weeks, list):
        print("\nRoadmap JSON is missing the 'weeks' array.")
        return False

    if len(weeks) != 12:
        print(
            f"\nRoadmap must contain exactly 12 weeks. "
            f"Received {len(weeks)}."
        )
        return False

    required_week_fields = {
        "weekNumber",
        "title",
        "status",
        "progressPercentage",
        "objectives",
        "resources",
        "practiceTasks",
        "completionCriteria"
    }

    for expected_number, week in enumerate(weeks, start=1):

        if not isinstance(week, dict):
            print(f"\nWeek {expected_number} is not an object.")
            return False

        missing = required_week_fields - set(week)

        if missing:
            print(
                f"\nWeek {expected_number} is missing: "
                + ", ".join(sorted(missing))
            )
            return False

        if week["weekNumber"] != expected_number:
            print(
                f"\nExpected week {expected_number}, "
                f"received {week['weekNumber']}."
            )
            return False

        if week["status"] != "not_started":
            week["status"] = "not_started"

        week["progressPercentage"] = 0

        if not isinstance(week["objectives"], list):
            return False

        if not isinstance(week["resources"], list):
            return False

        if not isinstance(week["practiceTasks"], list):
            return False

        if not isinstance(week["completionCriteria"], list):
            return False

        for task_index, task in enumerate(
            week["practiceTasks"],
            start=1
        ):

            if not isinstance(task, dict):
                print(
                    f"\nWeek {expected_number}, task "
                    f"{task_index} is invalid."
                )
                return False

            if not task.get("id"):
                task["id"] = (
                    f"w{expected_number}-t{task_index}"
                )

            if not task.get("title"):
                print(
                    f"\nWeek {expected_number}, task "
                    f"{task_index} has no title."
                )
                return False

            task["status"] = "not_started"

        # Normalize resource objects so the saved schema is predictable.
        normalized_resources = []

        for resource in week["resources"]:

            if isinstance(resource, str):
                normalized_resources.append(
                    {
                        "title": resource,
                        "url": "",
                        "type": "resource"
                    }
                )
                continue

            if isinstance(resource, dict):
                normalized_resources.append(
                    {
                        "title": str(
                            resource.get(
                                "title",
                                "Resource"
                            )
                        ),
                        "url": str(
                            resource.get(
                                "url",
                                ""
                            )
                        ),
                        "type": str(
                            resource.get(
                                "type",
                                "resource"
                            )
                        )
                    }
                )

        week["resources"] = normalized_resources

    canonical_roadmap = {
        "roadmap": {
            "targetRole": roadmap_data.get(
                "targetRole",
                context["learner_profile"].get(
                    "target_role",
                    "Senior Software Engineer"
                )
            ),
            "careerGoal": roadmap_data.get(
                "careerGoal",
                context["learner_profile"].get(
                    "career_goal",
                    ""
                )
            ),
            "timelineWeeks": 12,
            "currentSkills": roadmap_data.get(
                "currentSkills",
                []
            ),
            "priorityGaps": roadmap_data.get(
                "priorityGaps",
                []
            ),
            "weeks": weeks
        }
    }

    workflow_state["roadmap"] = canonical_roadmap

    roadmap_json = json.dumps(
        canonical_roadmap,
        indent=2,
        ensure_ascii=False
    )

    mcp_result = await session.call_tool(
        "save_learning_roadmap",
        arguments={
            "roadmap_json": roadmap_json
        }
    )

    extracted = extract_mcp_result(mcp_result)

    if tool_result_failed(
        mcp_result,
        extracted
    ):
        print("\nFailed to save roadmap:")

        if isinstance(extracted, dict):
            print(
                json.dumps(
                    extracted,
                    indent=2,
                    ensure_ascii=False
                )
            )
        else:
            print(extracted)

        return False

    print("\nRoadmap generated and saved successfully.")

    if isinstance(extracted, dict):

        saved_path = extracted.get("path")

        if saved_path:
            print(
                f"\nRoadmap file:\n{saved_path}"
            )

    return True


# ============================================================
# INITIALIZE PROGRESS
# ============================================================

async def initialize_progress(session):

    workflow_state["phase"] = "PROGRESS"

    roadmap_wrapper = workflow_state["roadmap"]

    if not isinstance(roadmap_wrapper, dict):
        print("\nInvalid roadmap state.")
        return False

    roadmap = roadmap_wrapper.get("roadmap")

    if not isinstance(roadmap, dict):
        print("\nRoadmap object missing.")
        return False

    weeks = roadmap.get("weeks", [])

    if not weeks:
        print("\nRoadmap contains no weeks.")
        return False

    first_week = weeks[0]

    week_number = first_week.get("weekNumber")
    topic = first_week.get("title", "Learning roadmap")

    if week_number is None:
        print("\nFirst week does not contain weekNumber.")
        return False

    mcp_result = await session.call_tool(
        "update_learning_progress",
        arguments={
            "week": week_number,
            "topic": topic,
            "status": "not_started",
            "notes": "Roadmap initialized."
        }
    )

    extracted = extract_mcp_result(mcp_result)

    if tool_result_failed(mcp_result, extracted):
        print("\nFailed to initialize progress:")
        print(extracted)
        return False

    workflow_state["progress_initialized"] = True

    print("\nProgress tracking initialized.")

    return True


# ============================================================
# FINAL RESPONSE
# ============================================================

def print_final_result():

    workflow_state["phase"] = "COMPLETE"
    workflow_state["completed"] = True

    profile = workflow_state[
        "learner_profile"
    ]

    gaps = workflow_state[
        "skill_gaps"
    ]

    roadmap = workflow_state[
        "roadmap"
    ]

    print("\n")
    print("=" * 70)
    print("EDUPATH LEARNING PLAN")
    print("=" * 70)

    print("\nLEARNER")

    print(
        profile["learner"]["name"]
    )

    print("\nTARGET ROLE")

    print(
        profile["target_role"]
    )

    print("\nCAREER GOAL")

    print(
        profile["career_goal"]
    )

    print("\nCURRENT SKILLS")

    for skill in profile["current_skills"]:

        if isinstance(skill, dict):

            print(
                f" - {skill.get('name')}: "
                f"{skill.get('evidence', '')}"
            )

        else:

            print(
                f" - {skill}"
            )

    print("\nSKILL GAPS")

    for gap in gaps.get(
        "gaps",
        []
    ):

        print(
            f" - {gap.get('skill')} "
            f"[{gap.get('priority')}]"
        )

        print(
            f"   {gap.get('gap')}"
        )

    print("\nROADMAP")

    roadmap_data = roadmap.get(
        "roadmap",
        {}
    )

    for week in roadmap_data.get(
        "weeks",
        []
    ):

        print("\n" + "-" * 60)

        print(
            f"Week {week.get('weekNumber')}: "
            f"{week.get('title')}"
        )

        print(
            f"Status: {week.get('status')}"
        )

        print(
            f"Progress: {week.get('progressPercentage')}%"
        )

        print("\nObjectives:")

        for item in week.get(
            "objectives",
            []
        ):
            print(f" - {item}")

        print("\nResources:")

        for resource in week.get(
            "resources",
            []
        ):

            if isinstance(resource, dict):
                print(
                    f" - {resource.get('title', 'Resource')}"
                )

                if resource.get("url"):
                    print(
                        f"   {resource['url']}"
                    )
            else:
                print(f" - {resource}")

        print("\nPractice Tasks:")

        for task in week.get(
            "practiceTasks",
            []
        ):

            if isinstance(task, dict):
                print(
                    f" - {task.get('title')} "
                    f"[{task.get('status')}]"
                )
            else:
                print(f" - {task}")

        print("\nCompletion Criteria:")

        for item in week.get(
            "completionCriteria",
            []
        ):
            print(f" - {item}")

    print("\n")
    print("=" * 70)
    print("EDUPATH WORKFLOW COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

async def main():

    client_directory = Path(
        __file__
    ).parent

    server_path = (
        client_directory
        / SERVER_RELATIVE_PATH
    )

    print(
        "\nMCP Server:"
    )

    print(
        server_path
    )

    if not server_path.exists():

        print(
            "\nERROR:"
        )

        print(
            f"MCP server not found: {server_path}"
        )

        return

    # --------------------------------------------------------
    # START MCP
    # --------------------------------------------------------

    server_params = StdioServerParameters(
        command="python",
        args=[
            str(server_path)
        ]
    )

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()

            print(
                "\nMCP connection established!"
            )

            # ------------------------------------------------
            # GET TOOLS
            # ------------------------------------------------

            mcp_response = (
                await session.list_tools()
            )

            all_tools = (
                mcp_response.tools
            )

            print(
                "\nAvailable MCP tools:"
            )

            for tool in all_tools:

                print(
                    f" - {tool.name}"
                )

            # ------------------------------------------------
            # ONLY SEND EDUPATH TOOLS TO QWEN
            # ------------------------------------------------

            edupath_tools = [

                tool
                for tool in all_tools
                if tool.name in ALLOWED_TOOLS

            ]

            ollama_tools = [

                convert_mcp_tool_to_ollama(
                    tool
                )

                for tool in edupath_tools
            ]

            print(
                "\nTools sent to Qwen:"
            )

            for tool in ollama_tools:

                print(
                    " -",
                    tool["function"]["name"]
                )

            # =================================================
            # USER CONVERSATION
            # =================================================

            while True:

                print(
                    "\n"
                )

                user_message = input(
                    "You: "
                ).strip()

                if not user_message:
                    continue

                if user_message.lower() == "/bye":
                    break

                # =================================================
                # NEW DOCUMENT
                # =================================================

                document_path = (
                    extract_document_path(
                        user_message
                    )
                )

                if document_path:

                    # Reset workflow for a new learner document.

                    workflow_state.clear()

                    workflow_state.update({

                        "phase":
                            "WAITING_FOR_DOCUMENT",

                        "document": {
                            "loaded": False,
                            "path": None,
                            "content": None
                        },

                        "resume_facts":
                            None,

                        "learner_input": {
                            "target_role": None,
                            "career_goal": None,
                            "skill_ratings": {},
                            "learning_preferences": {}
                        },

                        "learner_profile":
                            None,

                        "skill_gaps":
                            None,

                        "resources":
                            [],

                        "roadmap":
                            None,

                        "progress_initialized":
                            False,

                        "completed":
                            False
                    })

                    add_user_message(
                        user_message
                    )

                    # -----------------------------------------
                    # Tell Qwen exact path.
                    # -----------------------------------------

                    ConversationWindow.append({
                        "role": "user",
                        "content": (
                            "The learner document is located at:\n"
                            + document_path
                            + "\n\n"
                            "Read this exact file."
                        )
                    })

                    # -----------------------------------------
                    # STEP 1: READ DOCUMENT
                    # -----------------------------------------

                    success = (
                        await process_document_phase(
                            session,
                            ollama_tools
                        )
                    )

                    if not success:

                        print(
                            "\nCould not read the learner document."
                        )

                        continue

                    # -----------------------------------------
                    # STEP 2: EXTRACT RESUME FACTS
                    # -----------------------------------------

                    success = (
                        await extract_resume_facts()
                    )

                    if not success:

                        print(
                            "\nCould not extract learner facts."
                        )

                        continue

                    print(
                        "\n========================================"
                    )

                    print(
                        "Resume facts extracted"
                    )

                    print(
                        "========================================"
                    )

                    print(
                        json.dumps(
                            workflow_state[
                                "resume_facts"
                            ],
                            indent=2,
                            ensure_ascii=False
                        )
                    )

                    # -----------------------------------------
                    # STEP 3: ASK LEARNER
                    # -----------------------------------------

                    workflow_state[
                        "phase"
                    ] = "PROFILE_INFORMATION"

                    if not request_missing_profile_information():

                        print(
                            "\nTarget role and career goal are required."
                        )

                        continue

                    # -----------------------------------------
                    # STEP 4: BUILD + SAVE PROFILE
                    # -----------------------------------------

                    success = await save_profile(
                        session,
                        ollama_tools
                    )

                    if not success:
                        continue

                    # -----------------------------------------
                    # STEP 5: SKILL GAP
                    # -----------------------------------------

                    success = (
                        await create_skill_gap_analysis(
                            session,
                            ollama_tools
                        )
                    )

                    if not success:
                        continue

                    # -----------------------------------------
                    # STEP 6: RESOURCES
                    # -----------------------------------------

                    await research_resources(
                        session,
                        ollama_tools
                    )

                    # -----------------------------------------
                    # STEP 7: ROADMAP
                    # -----------------------------------------

                    success = await create_roadmap(
                        session
                    )

                    if not success:
                        continue

                    # -----------------------------------------
                    # STEP 8: PROGRESS
                    # -----------------------------------------

                    success = await initialize_progress(
                        session
                    )

                    if not success:
                        continue

                    # -----------------------------------------
                    # STEP 9: FINAL
                    # -----------------------------------------

                    print_final_result()

                    continue

                # =================================================
                # NORMAL CHAT / FOLLOW-UP
                # =================================================

                add_user_message(
                    user_message
                )

                # At this point you can later add:
                #
                # "I completed week 1"
                #
                # "I'm struggling with Spring Security"
                #
                # etc.
                #
                # Those should be handled by a separate
                # progress-update workflow instead of restarting
                # the resume workflow.

                print(
                    "\nEduPath:"
                )

                print(
                    "Please provide a learner document "
                    "(PDF/TXT) to start a learning plan."
                )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )