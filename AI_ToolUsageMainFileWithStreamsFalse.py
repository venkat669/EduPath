import asyncio
import json
import requests
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)

# You are an AI assistant. Use the available tools whenever they are appropriate.
# You: list the files in G:\PythonAI\MCP
MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/chat"

ConversationWindow = []



# Workflow States
workflow_state = {
    "profile_saved": False,
    "skill_gap_saved": False,
    "roadmap_saved": False,
    "progress_saved": False
}



# ============================================================
# MCP TOOL → OLLAMA TOOL FORMAT
# ============================================================


def convert_mcp_tool_to_ollama(tool):

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }



# ============================================================
# Function to extract the mcp results
# ============================================================


def extract_mcp_result(mcp_result):

    # --------------------------------------------------------
    # 1. Structured content exists
    # --------------------------------------------------------

    if mcp_result.structured_content is not None:
        return mcp_result.structured_content

    # --------------------------------------------------------
    # 2. Fall back to normal MCP content
    # --------------------------------------------------------

    if mcp_result.content:

        text_parts = []

        for item in mcp_result.content:

            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)

        if text_parts:

            combined_text = "\n".join(text_parts)

            # Try to convert JSON text into Python object
            try:
                return json.loads(combined_text)

            except json.JSONDecodeError:
                # It may simply be normal text
                return combined_text

    # --------------------------------------------------------
    # 3. Nothing usable returned
    # --------------------------------------------------------

    return {
        "error": "MCP tool returned no usable result"
    }




# ============================================================
# MESSAGE FUNCTIONS
# ============================================================


def add_user_message(message):

    ConversationWindow.append({"role": "user", "content": message})


def add_ai_message(message):

    ConversationWindow.append({"role": "assistant", "content": message})


def add_assistant_tool_calls(content, tool_calls):

    ConversationWindow.append(
        {"role": "assistant", "content": content or "", "tool_calls": tool_calls}
    )


def add_tool_message(tool_id, tool_name, message):

    ConversationWindow.append(
        {
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": json.dumps(message),
        }
    )


# ============================================================
# OLLAMA PAYLOAD
# ============================================================


def build_payload(system_prompt, tools):

    messages = []

    if system_prompt:

        messages.append({"role": "system", "content": system_prompt})

    messages.extend(ConversationWindow)

    return {"model": MODEL, "messages": messages, "tools": tools, "stream": False}


# ============================================================
# CALL QWEN / OLLAMA
# ============================================================


def prompt_ai(payload):

    response = requests.post(OLLAMA_URL, json=payload, stream=False)

    if response.status_code != 200:

        print(response.text)
        return None

    return response


# ============================================================
# MAIN
# ============================================================


async def main():

    # --------------------------------------------------------
    # Locate MCP Server
    # --------------------------------------------------------

    client_directory = Path(__file__).parent

    server_path = client_directory / "Servers" / "filesystem_server.py"

    print("\nMCP Server:")
    print(server_path)

    # --------------------------------------------------------
    # Configure MCP Server
    # --------------------------------------------------------

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
    )

    # --------------------------------------------------------
    # Start MCP Server
    # --------------------------------------------------------

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # ------------------------------------------------
            # Initialize MCP
            # ------------------------------------------------

            await session.initialize()

            print("\nMCP connection established!")

            # ------------------------------------------------
            # Get MCP Tools
            # ------------------------------------------------

            mcp_response = await session.list_tools()

            print("\nAvailable MCP Tools:")

            for tool in mcp_response.tools:

                print(" -", tool.name)

            # ------------------------------------------------
            # Convert MCP tools to Ollama format
            # ------------------------------------------------

            MCP_TOOLS = [
                convert_mcp_tool_to_ollama(tool) for tool in mcp_response.tools
            ]

            print("\nTools sent to Qwen:")

            for tool in MCP_TOOLS:

                

                print(" -", tool["function"]["name"])

            # ------------------------------------------------
            # System Prompt
            # ------------------------------------------------

            SystemPrompt = """ You are EduPath, an adaptive personalized learning agent.

Your purpose is to create evidence-based, personalized learning journeys for a learner based on:

- Current skills
- Actual skill proficiency
- Professional experience
- Projects
- Education
- Certifications
- Target role
- Career goals
- Identified skill gaps

You have access to MCP tools. You MUST use the appropriate tools instead of pretending that an action was completed.

============================================================
CORE RULES
============================================================

1. NEVER invent learner information.

2. NEVER invent:
   - Skill ratings
   - Experience
   - Projects
   - Certifications
   - Target roles
   - Career goals
   - Skill gaps
   - Learning progress

3. A skill appearing on a resume does NOT mean the learner is highly
   proficient in that skill.

4. A certification does NOT prove practical expertise.

5. Professional experience can be used as evidence that the learner has
   encountered or worked with a technology, but it must NOT automatically
   be converted into a 1-5 proficiency rating.

6. If the learner has not provided a proficiency rating, DO NOT guess one.

7. If important information is missing, ask the learner instead of making
   assumptions.

8. Do not create a generic curriculum.

9. Every roadmap must be based on evidence about this specific learner.

10. Do not provide the final roadmap until the required analysis and
    persistence steps have been completed.

============================================================
PROFICIENCY RATINGS
============================================================

When asking the learner to rate a skill, use this scale:

1 = Very basic / beginner
2 = Basic understanding
3 = Comfortable / intermediate
4 = Strong / advanced
5 = Expert / highly experienced

IMPORTANT:

The learner must provide the rating.

For example, if a resume says:

Python (Basic)

you may record:

Python = "Basic according to resume"

You MUST NOT convert this into:

Python = 2

unless the learner explicitly gives the rating 2.

Similarly, if the resume says:

Java Spring Boot

you may record that the learner has experience with Java Spring Boot,
but you MUST NOT automatically assign a rating of 4 or 5.

============================================================
PHASE 1 — ANALYZE LEARNER DOCUMENT
============================================================

If the user provides:

- Resume
- CV
- Portfolio
- Certificate
- Project description
- Learning document
- Other learner-related document

you MUST use:

read_document

to inspect it before creating the learner profile.

Extract only information supported by the document.

Separate information into:

- Demonstrated skills
- Self-described skills
- Professional experience
- Projects
- Education
- Certifications
- Uncertain areas
- Missing information

Do not invent information that is not present in the document.

============================================================
PHASE 2 — CHECK REQUIRED LEARNER INFORMATION
============================================================

Before creating the final roadmap, determine whether the following
information is available:

1. Target role
2. Career goal
3. Learner's skill proficiency ratings for important skills

If the target role is NOT explicitly known:

ASK THE LEARNER for their target role.

Do NOT infer the target role from the resume.

If the career goal is NOT explicitly known:

ASK THE LEARNER about their career goal.

Do NOT invent a career goal.

If important skill ratings are missing:

ASK the learner to provide the ratings.

Do not create the final roadmap until the necessary information is available.

If you need multiple pieces of information, ask for them together in a
clear questionnaire rather than asking one question at a time.

Example:

"Before I create your roadmap, I need:

Target role:
Career goal:

Rate these skills from 1-5:
Java:
Spring Boot:
Python:
SQL:
Docker:
AWS:"

Then WAIT for the learner's response.

============================================================
PHASE 3 — SAVE LEARNER PROFILE
============================================================

Once sufficient learner information is available:

Use:

save_learner_profile

The learner profile should contain information such as:

- Name
- Current skills
- Skill ratings provided by the learner
- Experience
- Projects
- Certifications
- Education
- Target role
- Career goals
- Uncertain areas

Do not invent missing values.

Use null, "unknown", or "not provided" where appropriate.

After calling save_learner_profile, verify that the tool returned a
successful result.

============================================================
PHASE 4 — SKILL GAP ANALYSIS
============================================================

After the learner profile has been saved:

Analyze the learner against their target role and career goal.

Identify:

1. Existing strengths
2. Required skills already demonstrated
3. Skills that require improvement
4. Missing skills
5. Uncertain skills
6. Important prerequisites

Distinguish between:

- Demonstrated knowledge
- Learner-reported proficiency
- Resume-listed knowledge
- Unverified knowledge
- Missing knowledge

Do not claim that the learner has a skill gap unless there is evidence
supporting that conclusion.

Then use:

save_skill_gap_analysis

to persist the analysis.

Do not merely describe the skill-gap analysis in your response.
Actually call the MCP tool.

============================================================
PHASE 5 — LEARNING RESOURCE RESEARCH
============================================================

For every important identified skill gap:

Use:

search_learning_resources

with an appropriate:

- topic
- level

The level must be based on the learner's actual proficiency and the
requirements of the target role.

Do not automatically choose "beginner" simply because a technology is
uncertain.

Use the learner's provided proficiency and diagnostic information.

The search results are INPUT DATA for constructing the roadmap.

IMPORTANT:

Do NOT stop after search_learning_resources.

Do NOT simply summarize the search results to the learner.

After receiving search results, continue the EduPath workflow.

Use read_web_page when you need additional information about a resource.

Prefer relevant resources over large lists of resources.

============================================================
PHASE 6 — CREATE LEARNING OBJECTIVES
============================================================

For every important skill gap, create specific learning objectives.

Objectives should be:

- Specific
- Measurable
- Relevant to the target role
- Appropriate for the learner's current level
- Practical

Avoid generic objectives such as:

"Learn Python."

Instead use objectives such as:

"Implement REST APIs in Python using FastAPI and explain request,
response, validation, and error-handling flows."

============================================================
PHASE 7 — CREATE PERSONALIZED ROADMAP
============================================================

Create a step-by-step weekly roadmap.

Each week MUST contain:

1. Week number
2. Focus area
3. Concepts
4. Learning objectives
5. Selected learning resources
6. Practice tasks
7. Project work
8. Completion criteria

The roadmap must connect the learner's:

CURRENT STATE
        ↓
SKILL GAPS
        ↓
LEARNING OBJECTIVES
        ↓
PRACTICE
        ↓
PROJECTS
        ↓
TARGET ROLE

Do not create a generic sequence of courses.

Use the learner's existing experience to avoid unnecessarily repeating
things they already know.

For example, if the learner already has Java Spring Boot experience,
the roadmap should build upon that experience when appropriate instead
of treating the learner as a completely new programmer.

============================================================
PHASE 8 — SAVE ROADMAP
============================================================

After creating the roadmap:

Use:

save_learning_roadmap

to persist it.

Do NOT claim that the roadmap has been saved unless the MCP tool
actually succeeds.

============================================================
PHASE 9 — TRACK PROGRESS
============================================================

After the roadmap has been successfully saved:

Initialize or update learner progress using:

update_learning_progress

Track:

- Current week
- Completed objectives
- In-progress objectives
- Pending objectives
- Difficulty or blockers if provided

Do not invent progress.

If the learner has not started the roadmap, do not claim that anything
has been completed.

============================================================
PHASE 10 — ADAPTIVE LEARNING
============================================================

If the learner later reports difficulty:

1. Identify the specific difficult topic.
2. Determine whether the problem is caused by a prerequisite gap.
3. Adjust the roadmap accordingly.
4. Add prerequisite learning if necessary.
5. Update the learner's progress.
6. Update the roadmap if required.

Do not simply tell the learner to study harder.

============================================================
CRITICAL WORKFLOW RULE
============================================================

The following sequence represents the normal EduPath workflow:

READ DOCUMENT
      ↓
CHECK REQUIRED INFORMATION
      ↓
ASK LEARNER FOR MISSING INFORMATION
      ↓
SAVE LEARNER PROFILE
      ↓
ANALYZE SKILL GAPS
      ↓
SAVE SKILL GAP ANALYSIS
      ↓
SEARCH LEARNING RESOURCES
      ↓
CREATE LEARNING OBJECTIVES
      ↓
CREATE WEEKLY ROADMAP
      ↓
SAVE LEARNING ROADMAP
      ↓
UPDATE LEARNING PROGRESS
      ↓
FINAL RESPONSE

IMPORTANT:

A search result is NOT a final answer.

A resource summary is NOT a final answer.

A learner profile is NOT a final answer.

A skill-gap analysis is NOT a final answer.

The final answer should contain the personalized roadmap only after the
required workflow has been completed.

============================================================
TOOL EXECUTION RULES
============================================================

When an MCP tool is required:

1. Call the tool.
2. Wait for its result.
3. Inspect the result.
4. Use the result in the next step.
5. Continue the workflow.

Never pretend that a tool was called.

Never pretend that a file was saved.

Never pretend that a roadmap was created or saved.

If a tool returns an error:

- Identify the error.
- Attempt a reasonable correction if possible.
- Do not claim success if the operation failed.

============================================================
FINAL RESPONSE
============================================================

Only provide the final roadmap when:

- Learner information is sufficient
- Learner profile has been saved
- Skill-gap analysis has been created and saved
- Relevant resources have been researched
- Roadmap has been created
- Roadmap has been saved
- Progress has been initialized or updated

The final response should include:

1. Learner summary
2. Target role
3. Current skill profile
4. Identified skill gaps
5. Learning objectives
6. Weekly roadmap
7. Projects
8. Completion criteria
9. How progress will be tracked

Keep the response clear and practical.

Do not dump every search result into the final response.

Select the resources that are most relevant to the learner.

============================================================
IMPORTANT FINAL RULE
============================================================

NEVER invent information to make the workflow appear complete.

If information is missing, ask the learner.

If a required MCP operation has not been completed, continue the
workflow rather than producing the final roadmap.
"""

            # ------------------------------------------------
            # Conversation
            # ------------------------------------------------

            while True:

                UserMsg = input("\nYou: ")

                if UserMsg == "/bye":
                    break

                add_user_message(UserMsg)

                # ============================================
                # ASK QWEN
                # ============================================

                payload = build_payload(SystemPrompt, MCP_TOOLS)

                AI_response = prompt_ai(payload)

                result = AI_response.json()

                print("\nQwen response:")
                print(result)

                # ============================================
                # TOOL LOOP
                # ============================================

                while True:

                    tool_calls = result.get("message", {}).get("tool_calls")

                    # ========================================================
                    # NO TOOL CALL
                    # ========================================================

                    if not tool_calls:

                        content = result["message"].get("content", "")

                        workflow_complete = (
                            workflow_state["profile_saved"]
                            and workflow_state["skill_gap_saved"]
                            and workflow_state["roadmap_saved"]
                            and workflow_state["progress_saved"]
                        )

                        # -----------------------------------------------
                        # Workflow is complete
                        # -----------------------------------------------

                        if workflow_complete:

                            add_ai_message(content)

                            print("\nAI:", content)

                            break

                        # -----------------------------------------------
                        # Workflow is NOT complete
                        # -----------------------------------------------

                        print("\nQwen stopped before completing the workflow.")

                        ConversationWindow.append({
                            "role": "user",
                            "content": (
                                "Do not finish yet. Continue the EduPath workflow. "
                                "Complete all required steps and use the appropriate MCP "
                                "tools before providing the final roadmap."
                            )
                        })

                        PAYLOAD = build_payload(SystemPrompt, MCP_TOOLS)

                        AI_RESPONSE = prompt_ai(PAYLOAD)

                        result = AI_RESPONSE.json()

                        continue

                    # ========================================================
                    # TOOL CALLS EXIST
                    # ========================================================

                    for tool in tool_calls:

                        tool_ID = tool["id"]

                        tool_Name = tool["function"]["name"]

                        tool_Arguments = tool["function"]["arguments"]

                        print("\nQwen requested MCP tool:")
                        print("Tool:", tool_Name)
                        print("Arguments:", tool_Arguments)

                        # ====================================================
                        # MCP CALL
                        # ====================================================

                        MCP_Result = await session.call_tool(
                            tool_Name,
                            arguments=tool_Arguments
                        )

                        print("\nMCP Result:")
                        print(MCP_Result)

                        # ====================================================
                        # UPDATE WORKFLOW STATE
                        # ====================================================

                        if tool_Name == "save_learner_profile":
                            workflow_state["profile_saved"] = True

                        elif tool_Name == "save_skill_gap_analysis":
                            workflow_state["skill_gap_saved"] = True

                        elif tool_Name == "save_learning_roadmap":
                            workflow_state["roadmap_saved"] = True

                        elif tool_Name == "update_learning_progress":
                            workflow_state["progress_saved"] = True

                        # ====================================================
                        # EXTRACT RESULT
                        # ====================================================

                        tool_result = extract_mcp_result(MCP_Result)

                        print("\nResult sent to Qwen:")
                        print(tool_result)

                        # ====================================================
                        # ADD TOOL RESULT TO CONVERSATION
                        # ====================================================

                        add_tool_message(
                            tool_ID,
                            tool_Name,
                            tool_result
                        )

                    # ========================================================
                    # ASK QWEN TO CONTINUE AFTER TOOL EXECUTION
                    # ========================================================

                    PAYLOAD = build_payload(SystemPrompt, MCP_TOOLS)

                    AI_RESPONSE = prompt_ai(PAYLOAD)

                    result = AI_RESPONSE.json()

                    print("\nQwen after MCP tool:")
                    print(result)


if __name__ == "__main__":

    asyncio.run(main())
