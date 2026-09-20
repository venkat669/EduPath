from mcp.server.mcpserver import MCPServer
import asyncio
from pathlib import Path
from pydantic import Field
from pypdf import PdfReader
import requests
from datetime import datetime, timedelta
from ddgs import DDGS
from bs4 import BeautifulSoup
import json


import shutil






HEADERS = {
    "User-Agent":
    "Mozilla/5.0"
}


def read_page(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        return text[:6000]

    except Exception as e:

        return str(e)


def web_search(query):

    with DDGS() as ddgs:

        if any(
            x in query.lower()
            for x in [
                "latest",
                "today",
                "news",
                "released",
                "recent"
            ]
        ):

            results = list(
                ddgs.news(
                    query,
                    region="in-en",
                    max_results=5
                )
            )

        else:

            results = list(
                ddgs.text(
                    query,
                    region="in-en",
                    max_results=3
                )
            )

    output = []

    for result in results:

        url = result.get(
            "href",
            result.get("url", "")
        )

        output.append({

            "title":
                result.get("title"),

            "snippet":
                result.get("body")
                or result.get("description"),

            "url":
                url,

            "page":
                read_page(url)

        })

    return output


# def web_search(query):
#     with DDGS() as ddgs:
#         return list(ddgs.text(query, max_results=5))


# ============================================================
# MCP SERVER
# ============================================================
#
# MCPServer creates the MCP server that exposes our Python
# functions as tools that an LLM can call.
#
# The name identifies our MCP server.
# ============================================================

# Create our MCP server
server = MCPServer(name="File System Server", version="1.0.0")


# ============================================================
# IMPORTANT: TOOL DESCRIPTION vs FIELD vs DOCSTRING
# ============================================================
#
# 1. @server.tool(description="...")
#
#    Describes WHAT THE MCP TOOL DOES.
#
#    Example:
#
#        description="Create a new text file."
#
#    This becomes part of the MCP tool definition and helps
#    the LLM understand when this tool should be used.
#
#
# 2. Field(description="...")
#
#    Describes WHAT EACH INPUT PARAMETER MEANS.
#
#    Example:
#
#        path: str = Field(
#            description="Path of the file to create."
#        )
#
#    "path: str" tells us the parameter is a string.
#
#    Field(...) adds additional metadata, such as the
#    description of that parameter.
#
#
# 3. Docstring
#
#        """
#        Creates a file.
#        """
#
#    This is normal Python documentation for humans reading
#    the source code.
#
#    Some MCP/framework implementations may also inspect
#    docstrings when generating tool metadata. However, when
#    we explicitly provide @server.tool(description=...), we
#    don't need to repeat the same tool description in the
#    docstring.
#
#    Therefore, in this file we use:
#
#        @server.tool(description=...)  -> TOOL description
#        Field(description=...)        -> INPUT description
#        Comments                       -> CODE explanation
#
# ============================================================


# ============================================================
# 1. CREATE FILE
# ============================================================


@server.tool(
    name="create_file", description="Create a new text file with the supplied content."
)
async def create_file(
    path: str = Field(description="Path where the new file should be created."),
    content: str = Field(description="Text content to write into the new file."),
) -> str:

    # Convert the string path into a Path object.
    file_path = Path(path)

    try:
        # write_text() creates the file and writes the content.
        #
        # If the file already exists, its contents are replaced.
        file_path.write_text(content, encoding="utf-8")

        return f"File created: {path}"

    except OSError as e:
        # Handle filesystem errors such as permission problems.
        return f"Could not create file '{path}': {e}"


# ============================================================
# 2. EDIT FILE
# ============================================================


@server.tool(
    name="edit_file", description="Replace the contents of an existing text file."
)
async def edit_file(
    path: str = Field(description="Path of the existing file to edit."),
    content: str = Field(
        description="New text content that replaces the existing content."
    ),
) -> str:

    # Convert the supplied string into a Path object.
    file_path = Path(path)

    try:
        # write_text() replaces all existing content in the file.
        file_path.write_text(content, encoding="utf-8")

        return f"File updated: {path}"

    except FileNotFoundError:
        # The requested file doesn't exist.
        return f"File not found: {path}"

    except OSError as e:
        # Handle other filesystem errors.
        return f"Could not edit file '{path}': {e}"


# ============================================================
# 3. APPEND TO FILE
# ============================================================


@server.tool(
    name="append_file",
    description="Append content to the end of an existing text file.",
)
async def append_file(
    path: str = Field(description="Path of the existing file to append to."),
    content: str = Field(description="Text content to add to the end of the file."),
) -> str:

    # Convert the string path into a Path object.
    file_path = Path(path)

    try:
        # "a" means append mode.
        #
        # Existing content is preserved and the new content
        # is added to the end of the file.
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(content)

        return f"Content appended to: {path}"

    except FileNotFoundError:
        return f"File not found: {path}"

    except OSError as e:
        return f"Could not append to file '{path}': {e}"


# ============================================================
# 4. DELETE FILE
# ============================================================


@server.tool(name="delete_file", description="Delete a file.")
async def delete_file(
    path: str = Field(description="Path of the file that should be deleted."),
) -> str:

    # Convert the string path into a Path object.
    file_path = Path(path)

    # Check whether the path exists.
    if not file_path.exists():
        return f"File does not exist: {path}"

    # Make sure the path points to a file and not a directory.
    if not file_path.is_file():
        return f"Path is not a file: {path}"

    try:
        # unlink() permanently removes the file.
        file_path.unlink()

        return f"File deleted: {path}"

    except OSError as e:
        return f"Could not delete file '{path}': {e}"


# ============================================================
# 5. LIST FILES
# ============================================================


@server.tool(name="list_files", description="List files in the specified directory.")
async def list_files(
    path: str = Field(description="Path of the directory that the files should be listed"),
) -> list[str]:

    workspace = Path(path)

    if not workspace.exists():
        return []

    if not workspace.is_dir():
        return []

    files = []

    for item in workspace.iterdir():

        if item.is_file():
            files.append(item.name)

    return files


# ============================================================
# 6. CHECK IF FILE EXISTS
# ============================================================


@server.tool(name="file_exists", description="Check whether a file exists.")
async def file_exists(
    path: str = Field(description="Path of the file to check."),
) -> bool:

    # is_file() returns True only when the path exists
    # and points to a file.
    return Path(path).is_file()


# ============================================================
# 7. CHECK PATH TYPE
# ============================================================


@server.tool(
    name="check_path", description="Check whether a path is a file or directory."
)
async def check_path(
    path: str = Field(description="Path to check."),
) -> str:

    # Convert the string into a Path object.
    file_path = Path(path)

    # Check whether the path points to a file.
    if file_path.is_file():
        return f"{path} is a file."

    # Check whether the path points to a directory.
    if file_path.is_dir():
        return f"{path} is a directory."

    # The path is neither a file nor a directory.
    return f"Path does not exist: {path}"


# ============================================================
# 8. CREATE DIRECTORY
# ============================================================


@server.tool(name="create_directory", description="Create a directory.")
async def create_directory(
    path: str = Field(description="Path of the directory to create."),
) -> str:

    # Convert the string path into a Path object.
    directory = Path(path)

    try:
        # parents=True:
        # Creates missing parent directories if necessary.
        #
        # exist_ok=True:
        # Does not raise an error if the directory already exists.
        directory.mkdir(parents=True, exist_ok=True)

        return f"Directory created: {path}"

    except OSError as e:
        return f"Could not create directory '{path}': {e}"


# ============================================================
# 9. RENAME FILE
# ============================================================


@server.tool(name="rename_file", description="Rename a file.")
async def rename_file(
    old_path: str = Field(description="Current path of the file to rename."),
    new_path: str = Field(description="New path or name for the file."),
) -> str:

    # Convert both paths into Path objects.
    source = Path(old_path)
    destination = Path(new_path)

    # Make sure the source exists.
    if not source.exists():
        return f"File does not exist: {old_path}"

    # Make sure the source is a file.
    if not source.is_file():
        return f"Source is not a file: {old_path}"

    try:
        # rename() changes the file's name/path.
        source.rename(destination)

        return f"Renamed {old_path} to {new_path}"

    except OSError as e:
        return f"Could not rename file: {e}"


# ============================================================
# 10. COPY FILE
# ============================================================


@server.tool(name="copy_file", description="Copy a file to another location.")
async def copy_file(
    source: str = Field(description="Path of the existing file to copy."),
    destination: str = Field(
        description="Path where the copied file should be created."
    ),
) -> str:

    # Convert both paths into Path objects.
    source_path = Path(source)
    destination_path = Path(destination)

    # Make sure the source file exists.
    if not source_path.is_file():
        return f"Source file does not exist: {source}"

    try:
        # copy2() copies the file and preserves file metadata.
        shutil.copy2(source_path, destination_path)

        return f"Copied {source} to {destination}"

    except OSError as e:
        return f"Could not copy file: {e}"


# ============================================================
# 11. MOVE FILE
# ============================================================


@server.tool(name="move_file", description="Move a file to another location.")
async def move_file(
    source: str = Field(description="Current path of the file to move."),
    destination: str = Field(
        description="Destination path where the file should be moved."
    ),
) -> str:

    # Convert both paths into Path objects.
    source_path = Path(source)
    destination_path = Path(destination)

    # Make sure the source file exists.
    if not source_path.is_file():
        return f"Source file does not exist: {source}"

    try:
        # move() moves the file to the destination.
        shutil.move(source_path, destination_path)

        return f"Moved {source} to {destination}"

    except OSError as e:
        return f"Could not move file: {e}"


    

# ============================================================
# 12. VIEW FILE CONTENTS
# ============================================================


@server.tool(
    name="view_Text_file", description="Read and return the contents of a text file."
)
async def view_file(
    path: str = Field(
        description="Path of the text file whose contents should be read."
    ),
) -> str:

    # Convert the string path into a Path object.
    file_path = Path(path)

    # Make sure the path exists and points to a file.
    if not file_path.is_file():
        return f"File does not exist: {path}"

    try:
        # read_text() opens the file, reads all its contents,
        # and returns the contents as a string.
        return file_path.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        # The file exists but is not valid UTF-8 text.
        return f"Could not read '{path}': file is not a valid UTF-8 text file."

    except OSError as e:
        # Handle other filesystem errors such as permission issues.
        return f"Could not read file '{path}': {e}"


@server.tool()
async def sample_Tool_Test() -> list[str]:
    """
    this is for testing that whern asked for "List Tools" from client side, The MCP Server simply returns a list of all python functions that has "@server.tool" above its async function definition , with discritpion as the docs
    
    """


# ============================================================
# 13. VIEW PDF_FILE CONTENTS
# ============================================================


@server.tool(
    name="read_pdf_file",
    description="Read and extract all text from a PDF document. If the given file is of "".pdf"" format , this toll is used"
)
async def read_pdf_file(
    path: str = Field(
        description="Path to the PDF document to read."
    )
) -> str:

    file_path = Path(path)

    if not file_path.is_file():
        return f"PDF file does not exist: {path}"

    if file_path.suffix.lower() != ".pdf":
        return f"File is not a PDF: {path}"

    try:

        reader = PdfReader(str(file_path))

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:
                pages.append(
                    f"\n--- PAGE {page_number} ---\n{text}"
                )

        if not pages:
            return (
                "The PDF was opened successfully, "
                "but no extractable text was found."
            )

        return "\n".join(pages)

    except Exception as e:

        return f"Could not read PDF '{path}': {e}"


@server.tool(
    name="read_document",
    description=(
        "Read a learner document and extract its text. "
        "Supports TXT and PDF files. Use this when you need "
        "to understand a resume, portfolio, certificate, "
        "project description, notes, or other learning document."
    )
)
async def read_document(
    path: str = Field(
        description="Path to the PDF or TXT document to read."
    )
) -> str:

    file_path = Path(path)

    if not file_path.is_file():
        return f"Document does not exist: {path}"

    extension = file_path.suffix.lower()

    try:

        if extension == ".txt":

            return file_path.read_text(
                encoding="utf-8"
            )

        elif extension == ".pdf":

            reader = PdfReader(str(file_path))

            pages = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text:
                    pages.append(
                        f"\n--- PAGE {page_number} ---\n{text}"
                    )

            return "\n".join(pages)

        else:

            return (
                f"Unsupported document type: {extension}. "
                "Supported types: PDF, TXT."
            )

    except Exception as e:

        return f"Could not read document '{path}': {e}"




@server.tool(
    name="save_learner_profile",
    description=(
        "Save the learner's current profile including skills, "
        "experience, target role, career goal, projects, "
        "known topics, weak topics, and uncertain topics."
    )
)
async def save_learner_profile(
    profile_json: str = Field(
        description="Learner profile as a JSON string."
    )
) -> str:

    try:

        data_directory = Path("data")
        data_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        profile_path = (
            data_directory /
            "learner_profile.json"
        )

        profile_path.write_text(
            profile_json,
            encoding="utf-8"
        )

        return "Learner profile saved successfully."

    except Exception as e:

        return f"Could not save learner profile: {e}"






@server.tool(
    name="get_learner_profile",
    description=(
        "Retrieve the learner's saved profile. "
        "Use this when you need the learner's current "
        "skills, experience, goals, or known knowledge."
    )
)
async def get_learner_profile() -> str:

    profile_path = Path(
        "data/learner_profile.json"
    )

    if not profile_path.exists():

        return "No learner profile has been created yet."

    try:

        return profile_path.read_text(
            encoding="utf-8"
        )

    except Exception as e:

        return f"Could not read learner profile: {e}"





@server.tool(
    name="save_skill_gap_analysis",
    description=(
        "Save the identified skill gaps between the learner's "
        "current capabilities and target role."
    )
)
async def save_skill_gap_analysis(
    gap_analysis_json: str = Field(
        description="Skill gap analysis as JSON."
    )
) -> str:

    try:

        data_directory = Path("data")
        data_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        path = (
            data_directory /
            "skill_gaps.json"
        )

        path.write_text(
            gap_analysis_json,
            encoding="utf-8"
        )

        return "Skill gap analysis saved successfully."

    except Exception as e:

        return f"Could not save skill gaps: {e}"


@server.tool(
    name="get_skill_gap_analysis",
    description="Retrieve the learner's current skill gap analysis."
)
async def get_skill_gap_analysis() -> str:

    path = Path(
        "data/skill_gaps.json"
    )

    if not path.exists():
        return "No skill gap analysis exists yet."

    return path.read_text(
        encoding="utf-8"
    )



@server.tool(
    name="save_learning_roadmap",
    description=(
        "Save the learner's personalized learning roadmap "
        "containing weekly objectives, topics, resources, "
        "practice tasks, projects, and completion criteria."
    )
)
async def save_learning_roadmap(
    roadmap_json: str = Field(
        description="Personalized roadmap as JSON."
    )
) -> str:

    try:

        data_directory = Path("data")
        data_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        roadmap_path = (
            data_directory /
            "learning_roadmap.json"
        )

        roadmap_path.write_text(
            roadmap_json,
            encoding="utf-8"
        )

        return "Learning roadmap saved successfully."

    except Exception as e:

        return f"Could not save roadmap: {e}"



@server.tool(
    name="get_learning_roadmap",
    description="Retrieve the learner's current personalized roadmap."
)
async def get_learning_roadmap() -> str:

    roadmap_path = (
        Path("data") /
        "learning_roadmap.json"
    )

    if not roadmap_path.exists():
        return "No learning roadmap exists yet."

    return roadmap_path.read_text(
        encoding="utf-8"
    )

@server.tool(
    name="update_learning_progress",
    description=(
        "Update the learner's progress for a week, objective, "
        "practice task, or project."
    )
)
async def update_learning_progress(
    week: int = Field(
        description="Week number being updated."
    ),
    topic: str = Field(
        description="Topic or objective being updated."
    ),
    status: str = Field(
        description=(
            "Progress status: not_started, in_progress, "
            "completed, or struggling."
        )
    ),
    notes: str = Field(
        default="",
        description="Learner's notes about their progress or difficulties."
    )
) -> str:

    import json

    DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    progress_path = DATA_DIRECTORY / "progress.json"

    # --------------------------------------------------
    # Load existing progress
    # --------------------------------------------------

    if progress_path.exists():

        try:
            progress = json.loads(
                progress_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError:

            progress = {
                "weeks": {}
            }

    else:

        progress = {
            "weeks": {}
        }

    # --------------------------------------------------
    # Make sure "weeks" exists
    # --------------------------------------------------

    if "weeks" not in progress:

        progress["weeks"] = {}

    # --------------------------------------------------
    # Get current week
    # --------------------------------------------------

    week_key = str(week)

    if week_key not in progress["weeks"]:

        progress["weeks"][week_key] = {
            "topics": {}
        }

    # --------------------------------------------------
    # Update topic
    # --------------------------------------------------

    progress["weeks"][week_key]["topics"][topic] = {
        "status": status,
        "notes": notes
    }

    # --------------------------------------------------
    # Save progress
    # --------------------------------------------------

    progress_path.write_text(
        json.dumps(
            progress,
            indent=2
        ),
        encoding="utf-8"
    )

    return (
        f"Progress updated: Week {week}, "
        f"{topic} -> {status}"
    )

@server.tool(
    name="get_learning_progress",
    description=(
        "Retrieve the learner's current learning progress, "
        "including completed and struggling topics."
    )
)
async def get_learning_progress() -> str:

    progress_path = (
        Path("data") /
        "progress.json"
    )

    if not progress_path.exists():
        return "No learning progress has been recorded."

    return progress_path.read_text(
        encoding="utf-8"
    )


@server.tool(
    name="search_learning_resources",
    description=(
        "Search the web for learning resources for a specific "
        "skill or topic. Finds tutorials, documentation, courses, "
        "practice material, and project resources."
    )
)
async def search_learning_resources(
    topic: str = Field(
        description="The skill or topic to find learning material for."
    ),
    level: str = Field(
        default="beginner",
        description="Learner level: beginner, intermediate, or advanced."
    )
) -> list:

    queries = [
        f"{topic} {level} tutorial",
        f"{topic} official documentation",
        f"{topic} {level} course",
        f"{topic} practice exercises",
        f"{topic} hands on project"
    ]

    results = []

    try:

        with DDGS() as ddgs:

            for query in queries:

                search_results = ddgs.text(
                    query,
                    max_results=5
                )

                for result in search_results:

                    results.append({
                        "title": result.get("title"),
                        "url": result.get("href"),
                        "description": result.get("body")
                    })

        return results

    except Exception as e:

        return [
            {
                "error": f"Web search failed: {e}"
            }
        ]



@server.tool(
    name="read_web_page",
    description=(
        "Read the text content of a web page. "
        "Use this after finding a learning resource when "
        "you need to inspect its actual content."
    )
)
async def read_web_page(
    url: str = Field(
        description="URL of the web page to read."
    )
) -> str:

    # Put your existing BeautifulSoup read_page()
    # implementation here.

    return read_page(url)


# ============================================================
# START MCP SERVER
# ============================================================
#
# This starts the MCP server when this Python file is executed
# directly.
# ============================================================

if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())




