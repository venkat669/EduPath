import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

# Optional web search
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

# PDF support
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ============================================================
# MCP SERVER
# ============================================================

server = MCPServer(
    name="EduPath Learning Server",
    version="1.0.0"
)


# ============================================================
# PATHS
# ============================================================

SERVER_DIR = Path(__file__).resolve().parent

DATA_DIR = SERVER_DIR / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LEARNER_PROFILE_FILE = (
    DATA_DIR / "learner_profile.json"
)

SKILL_GAPS_FILE = (
    DATA_DIR / "skill_gaps.json"
)

LEARNING_ROADMAP_FILE = (
    DATA_DIR / "learning_roadmap.json"
)

PROGRESS_FILE = (
    DATA_DIR / "progress.json"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def success_result(
    message: str,
    **extra
):
    result = {
        "success": True,
        "message": message
    }

    result.update(extra)

    return result


def error_result(
    message: str,
    **extra
):
    result = {
        "error": message
    }

    result.update(extra)

    return result


def load_json_file(
    file_path: Path
):

    if not file_path.exists():

        return None

    try:

        return json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError:

        return None


def save_json_file(
    file_path: Path,
    data: Any
):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# DOCUMENT READING
# ============================================================

def extract_pdf_text(
    file_path: Path
):

    if PdfReader is None:

        raise RuntimeError(
            "pypdf is not installed. "
            "Install it with: pip install pypdf"
        )

    reader = PdfReader(
        str(file_path)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def extract_text_file(
    file_path: Path
):

    return file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# TOOL 1
# READ DOCUMENT
# ============================================================

@server.tool()
async def read_document(
    file_path: str
):
    """
    Read a learner document.

    Supported formats:
    - PDF
    - TXT

    Returns the complete extracted document text.

    This tool should be used when EduPath receives a learner
    resume or learning document.
    """

    if not file_path:

        return error_result(
            "file_path is required."
        )

    path = Path(
        file_path
    )

    if not path.exists():

        return error_result(
            "Document does not exist.",
            path=str(path)
        )

    if not path.is_file():

        return error_result(
            "The provided path is not a file.",
            path=str(path)
        )

    extension = (
        path.suffix.lower()
    )

    try:

        if extension == ".pdf":

            text = extract_pdf_text(
                path
            )

        elif extension == ".txt":

            text = extract_text_file(
                path
            )

        else:

            return error_result(
                "Unsupported document format.",
                supported_formats=[
                    ".pdf",
                    ".txt"
                ],
                received_format=extension
            )

    except Exception as exc:

        return error_result(
            "Failed to read document.",
            details=str(exc)
        )

    if not text.strip():

        return error_result(
            "The document was read but no text was extracted.",
            path=str(path)
        )

    return {
        "success": True,
        "path": str(path),
        "format": extension,
        "text": text,
        "character_count": len(text)
    }


# ============================================================
# PROFILE VALIDATION
# ============================================================

ALLOWED_PROFILE_FIELDS = {
    "learner",
    "current_skills",
    "experience",
    "projects",
    "target_role",
    "career_goal",
    "skill_ratings",
    "learning_preferences"
}


def validate_learner_profile(
    profile: Any
):

    if not isinstance(
        profile,
        dict
    ):

        return (
            False,
            "Profile must be a JSON object."
        )

    # --------------------------------------------------------
    # Prevent arbitrary information from being saved.
    # --------------------------------------------------------

    unknown_fields = (
        set(profile.keys())
        - ALLOWED_PROFILE_FIELDS
    )

    if unknown_fields:

        return (
            False,
            "Profile contains unsupported fields: "
            + ", ".join(
                sorted(unknown_fields)
            )
        )

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

    missing = (
        required_fields
        - set(profile.keys())
    )

    if missing:

        return (
            False,
            "Profile is missing required fields: "
            + ", ".join(
                sorted(missing)
            )
        )

    if not isinstance(
        profile["learner"],
        dict
    ):

        return (
            False,
            "'learner' must be an object."
        )

    if not isinstance(
        profile["current_skills"],
        list
    ):

        return (
            False,
            "'current_skills' must be a list."
        )

    if not isinstance(
        profile["experience"],
        list
    ):

        return (
            False,
            "'experience' must be a list."
        )

    if not isinstance(
        profile["projects"],
        list
    ):

        return (
            False,
            "'projects' must be a list."
        )

    if not isinstance(
        profile["skill_ratings"],
        dict
    ):

        return (
            False,
            "'skill_ratings' must be an object."
        )

    if not isinstance(
        profile["learning_preferences"],
        dict
    ):

        return (
            False,
            "'learning_preferences' must be an object."
        )

    if not profile.get(
        "target_role"
    ):

        return (
            False,
            "target_role is required."
        )

    if not profile.get(
        "career_goal"
    ):

        return (
            False,
            "career_goal is required."
        )

    return (
        True,
        None
    )


# ============================================================
# TOOL 2
# SAVE LEARNER PROFILE
# ============================================================

@server.tool()
async def save_learner_profile(
    profile_json: str
):
    """
    Save the concentrated learner profile.

    The profile must be valid JSON.

    The profile contains only:
    - learner
    - current_skills
    - experience
    - projects
    - target_role
    - career_goal
    - skill_ratings
    - learning_preferences

    AI-derived skill gaps and learning roadmaps must NOT be
    stored inside the learner profile.
    """

    if not profile_json:

        return error_result(
            "profile_json is required."
        )

    try:

        profile = json.loads(
            profile_json
        )

    except json.JSONDecodeError as exc:

        return error_result(
            "profile_json is not valid JSON.",
            details=str(exc)
        )

    valid, validation_error = (
        validate_learner_profile(
            profile
        )
    )

    if not valid:

        return error_result(
            validation_error
        )

    try:

        save_json_file(
            LEARNER_PROFILE_FILE,
            profile
        )

    except Exception as exc:

        return error_result(
            "Failed to save learner profile.",
            details=str(exc)
        )

    return success_result(
        "Learner profile saved successfully.",
        path=str(
            LEARNER_PROFILE_FILE
        )
    )


# ============================================================
# TOOL 3
# GET LEARNER PROFILE
# ============================================================

@server.tool()
async def get_learner_profile():
    """
    Retrieve the saved learner profile.
    """

    profile = load_json_file(
        LEARNER_PROFILE_FILE
    )

    if profile is None:

        return error_result(
            "Learner profile does not exist."
        )

    return {
        "success": True,
        "profile": profile
    }


# ============================================================
# SKILL GAP VALIDATION
# ============================================================

def validate_skill_gap_analysis(
    data
):

    if not isinstance(
        data,
        dict
    ):

        return (
            False,
            "Skill-gap analysis must be an object."
        )

    if not isinstance(
        data.get("gaps"),
        list
    ):

        return (
            False,
            "'gaps' must be a list."
        )

    for gap in data["gaps"]:

        if not isinstance(
            gap,
            dict
        ):

            return (
                False,
                "Each gap must be an object."
            )

        required = {
            "skill",
            "current_evidence",
            "required_for_role",
            "gap",
            "priority"
        }

        missing = (
            required
            - set(gap.keys())
        )

        if missing:

            return (
                False,
                "Skill gap is missing fields: "
                + ", ".join(
                    sorted(missing)
                )
            )

        if gap["priority"] not in {
            "high",
            "medium",
            "low"
        }:

            return (
                False,
                "Gap priority must be high, medium or low."
            )

    return (
        True,
        None
    )


# ============================================================
# TOOL 4
# SAVE SKILL GAP ANALYSIS
# ============================================================

@server.tool()
async def save_skill_gap_analysis(
    gap_analysis_json: str
):
    """
    Save AI-derived skill gap analysis.

    This file contains analysis, not learner facts.
    """

    if not gap_analysis_json:

        return error_result(
            "gap_analysis_json is required."
        )

    try:

        data = json.loads(
            gap_analysis_json
        )

    except json.JSONDecodeError as exc:

        return error_result(
            "gap_analysis_json is not valid JSON.",
            details=str(exc)
        )

    valid, validation_error = (
        validate_skill_gap_analysis(
            data
        )
    )

    if not valid:

        return error_result(
            validation_error
        )

    try:

        save_json_file(
            SKILL_GAPS_FILE,
            data
        )

    except Exception as exc:

        return error_result(
            "Failed to save skill-gap analysis.",
            details=str(exc)
        )

    return success_result(
        "Skill-gap analysis saved successfully.",
        path=str(
            SKILL_GAPS_FILE
        )
    )


# ============================================================
# TOOL 5
# GET SKILL GAP ANALYSIS
# ============================================================

@server.tool()
async def get_skill_gap_analysis():
    """
    Retrieve the saved skill-gap analysis.
    """

    data = load_json_file(
        SKILL_GAPS_FILE
    )

    if data is None:

        return error_result(
            "Skill-gap analysis does not exist."
        )

    return {
        "success": True,
        "skill_gaps": data
    }


# ============================================================
# ROADMAP VALIDATION
# ============================================================

def validate_learning_roadmap(roadmap):

    if not isinstance(roadmap, dict):
        return False, "Roadmap must be an object."

    if "roadmap" not in roadmap:
        return False, "Missing 'roadmap' field."

    data = roadmap["roadmap"]

    if not isinstance(data, dict):
        return False, "'roadmap' must be an object."

    required_fields = {
        "targetRole",
        "careerGoal",
        "timelineWeeks",
        "currentSkills",
        "priorityGaps",
        "weeks"
    }

    missing = required_fields - set(data.keys())

    if missing:
        return (
            False,
            "Roadmap is missing fields: "
            + ", ".join(sorted(missing))
        )

    if data["timelineWeeks"] != 12:
        return False, "timelineWeeks must be 12."

    weeks = data["weeks"]

    if not isinstance(weeks, list):
        return False, "'weeks' must be a list."

    if len(weeks) != 12:
        return (
            False,
            f"Roadmap must contain 12 weeks. Received {len(weeks)}."
        )

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
            return False, f"Week {expected_number} must be an object."

        missing_week_fields = (
            required_week_fields - set(week.keys())
        )

        if missing_week_fields:
            return (
                False,
                f"Week {expected_number} is missing: "
                + ", ".join(sorted(missing_week_fields))
            )

        if week["weekNumber"] != expected_number:
            return (
                False,
                f"Expected week {expected_number}, "
                f"received {week['weekNumber']}."
            )

        if week["status"] != "not_started":
            return (
                False,
                f"Week {expected_number} must initially "
                f"be 'not_started'."
            )

        if week["progressPercentage"] != 0:
            return (
                False,
                f"Week {expected_number} must initially "
                f"have progressPercentage 0."
            )

    return True, None


# ============================================================
# TOOL 6
# SAVE LEARNING ROADMAP
# ============================================================

# @server.tool()
# async def save_learning_roadmap(
#     roadmap_json: str
# ):
#     """
#     Save the AI-generated personalized learning roadmap.
#     """

#     if not roadmap_json:

#         return error_result(
#             "roadmap_json is required."
#         )

#     try:

#         roadmap = json.loads(
#             roadmap_json
#         )

#     except json.JSONDecodeError as exc:

#         return error_result(
#             "roadmap_json is not valid JSON.",
#             details=str(exc)
#         )

#     valid, validation_error = (
#         validate_learning_roadmap(
#             roadmap
#         )
#     )

#     if not valid:

#         return error_result(
#             validation_error
#         )

#     try:

#         save_json_file(
#             LEARNING_ROADMAP_FILE,
#             roadmap
#         )

#     except Exception as exc:

#         return error_result(
#             "Failed to save learning roadmap.",
#             details=str(exc)
#         )

#     return success_result(
#         "Learning roadmap saved successfully.",
#         path=str(
#             LEARNING_ROADMAP_FILE
#         )
#     )

@server.tool()
async def save_learning_roadmap(
    roadmap_json: str
):
    """
    Save the learner's personalized learning roadmap.

    Expected structure:

    {
        "roadmap": {
            "targetRole": "...",
            "careerGoal": "...",
            "timelineWeeks": 12,
            "currentSkills": [],
            "priorityGaps": [],
            "weeks": []
        }
    }
    """

    if not roadmap_json:
        return error_result("roadmap_json is required.")

    try:
        roadmap = json.loads(roadmap_json)
    except json.JSONDecodeError as exc:
        return error_result(
            "roadmap_json is not valid JSON.",
            details=str(exc)
        )

    valid, validation_error = validate_learning_roadmap(roadmap)

    if not valid:
        return error_result(validation_error)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    roadmap_path = (
        DATA_DIR / "learning_roadmap.json"
    ).resolve()

    try:
        roadmap_path.write_text(
            json.dumps(
                roadmap,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )
    except Exception as exc:
        return error_result(
            "Failed to save learning roadmap.",
            details=str(exc),
            path=str(roadmap_path)
        )

    print(
        f"\n[EduPath] Learning roadmap saved to:\n"
        f"{roadmap_path}"
    )

    return success_result(
        "Learning roadmap saved successfully.",
        path=str(roadmap_path),
        week_count=len(
            roadmap["roadmap"]["weeks"]
        )
    )


# ============================================================
# TOOL 7
# GET LEARNING ROADMAP
# ============================================================

@server.tool()
async def get_learning_roadmap():
    """
    Retrieve the saved learning roadmap.
    """

    roadmap = load_json_file(
        LEARNING_ROADMAP_FILE
    )

    if roadmap is None:

        return error_result(
            "Learning roadmap does not exist."
        )

    return {
        "success": True,
        "roadmap": roadmap
    }


# ============================================================
# PROGRESS
# ============================================================

VALID_PROGRESS_STATUSES = {
    "not_started",
    "in_progress",
    "completed",
    "struggling"
}


def load_progress():

    data = load_json_file(
        PROGRESS_FILE
    )

    if data is None:

        return {
            "weeks": {}
        }

    if not isinstance(
        data,
        dict
    ):

        return {
            "weeks": {}
        }

    if "weeks" not in data:

        data["weeks"] = {}

    return data


# ============================================================
# TOOL 8
# UPDATE LEARNING PROGRESS
# ============================================================

@server.tool()
async def update_learning_progress(
    week: int,
    topic: str,
    status: str,
    notes: str = ""
):
    """
    Update actual learner progress.

    Valid statuses:
    - not_started
    - in_progress
    - completed
    - struggling
    """

    if week < 1:

        return error_result(
            "week must be greater than 0."
        )

    if not topic:

        return error_result(
            "topic is required."
        )

    if status not in VALID_PROGRESS_STATUSES:

        return error_result(
            "Invalid progress status.",
            valid_statuses=list(
                VALID_PROGRESS_STATUSES
            )
        )

    progress = load_progress()

    week_key = str(
        week
    )

    progress["weeks"][week_key] = {

        "week": week,

        "topic": topic,

        "status": status,

        "notes": notes
    }

    try:

        save_json_file(
            PROGRESS_FILE,
            progress
        )

    except Exception as exc:

        return error_result(
            "Failed to update learning progress.",
            details=str(exc)
        )

    return success_result(
        "Learning progress updated successfully.",
        week=week,
        topic=topic,
        status=status
    )


# ============================================================
# TOOL 9
# GET LEARNING PROGRESS
# ============================================================

@server.tool()
async def get_learning_progress():
    """
    Retrieve learner progress.
    """

    progress = load_progress()

    return {
        "success": True,
        "progress": progress
    }


# ============================================================
# RESOURCE SEARCH
# ============================================================

@server.tool()
async def search_learning_resources(
    topic: str,
    level: str
):
    """
    Search for learning resources.

    Returns a small structured collection of resources.

    This tool searches for educational resources only.
    """

    if not topic:

        return error_result(
            "topic is required."
        )

    if not level:

        return error_result(
            "level is required."
        )

    if DDGS is None:

        return error_result(
            "ddgs is not installed.",
            install_command="pip install ddgs"
        )

    queries = [

        f"{topic} {level} official documentation",

        f"{topic} {level} tutorial",

        f"{topic} {level} hands on project"
    ]

    resources = []

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:

                    results = list(
                        ddgs.text(
                            query,
                            max_results=3
                        )
                    )

                except Exception:
                    continue

                for item in results:

                    title = (
                        item.get("title")
                        or ""
                    )

                    url = (
                        item.get("href")
                        or item.get("url")
                        or ""
                    )

                    description = (
                        item.get("body")
                        or ""
                    )

                    if not url:
                        continue

                    resources.append({

                        "title": title,

                        "url": url,

                        "description":
                            description[:500]
                    })

                    if len(resources) >= 8:
                        break

                if len(resources) >= 8:
                    break

    except Exception as exc:

        return error_result(
            "Web search failed.",
            details=str(exc)
        )

    if not resources:

        return error_result(
            "No learning resources found.",
            topic=topic,
            level=level
        )

    # Remove duplicate URLs.
    unique = {}

    for resource in resources:

        unique[
            resource["url"]
        ] = resource

    resources = list(
        unique.values()
    )

    return {

        "success": True,

        "topic": topic,

        "level": level,

        "resources": resources[:8]
    }


# ============================================================
# WEB PAGE READER
# ============================================================

@server.tool()
async def read_web_page(
    url: str
):
    """
    Read the text from a web page.

    Used when EduPath needs to inspect a resource returned
    by search_learning_resources.
    """

    if not url:

        return error_result(
            "url is required."
        )

    try:

        import requests
        from bs4 import BeautifulSoup

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent":
                    "EduPath/1.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove non-content elements.
        for tag in soup(
            [
                "script",
                "style",
                "noscript"
            ]
        ):

            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if not text:

            return error_result(
                "No readable text found on page."
            )

        return {

            "success": True,

            "url": url,

            "text": text[:20000]
        }

    except Exception as exc:

        return error_result(
            "Failed to read web page.",
            details=str(exc)
        )


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    print(
        "Starting EduPath MCP Server..."
    )

    print(
        f"Data directory: {DATA_DIR}"
    )

    server.run(
        transport="stdio"
    )