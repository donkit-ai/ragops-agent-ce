import warnings

# Suppress all warnings immediately, before any other imports
warnings.filterwarnings("ignore")
# Suppress warnings from importlib bootstrap (SWIG-related)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
# Suppress all DeprecationWarnings globally
warnings.simplefilter("ignore", DeprecationWarning)
import json
import os
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional

from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field


class ChecklistItem(BaseModel):
    """Single checklist task item."""

    id: str = Field(
        ..., description="Stable item identifier (e.g., 'item_0'). Used to update status later."
    )
    description: str = Field(
        ..., description="Human-readable task description (what needs to be done)."
    )
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending",
        description="Item status: one of 'pending', 'in_progress', 'completed', 'failed'.",
    )


class Checklist(BaseModel):
    """Checklist with a name and ordered list of items."""

    name: str = Field(
        ..., description="Unique checklist name; used as the JSON filename without extension."
    )
    items: List[ChecklistItem] = Field(
        ..., description="Ordered list of checklist items. IDs are stable across updates."
    )


server = FastMCP(
    "rag-checklist",
)


def _save_checklist(checklist: Checklist) -> str:
    output_dir = Path("./ragops_checklists")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{checklist.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checklist.model_dump(), f, indent=2)
    return str(output_path)


def _load_checklist(name: str) -> Optional[Checklist]:
    output_path = Path("./ragops_checklists") / f"{name}.json"
    if not output_path.exists():
        return None
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
        return Checklist(**data)


# ========= Pydantic argument schemas =========
class CreateChecklistArgs(BaseModel):
    """Arguments for create_checklist tool."""

    name: str = Field(..., description="Unique checklist name. Will be used as '<name>.json' file.")
    items: List[str] = Field(
        ..., description="List of task descriptions. IDs will be auto-generated as 'item_<index>'."
    )


class GetChecklistArgs(BaseModel):
    """Arguments for get_checklist tool."""

    name: str = Field(..., description="Checklist name to load (without '.json').")


class UpdateChecklistItemArgs(BaseModel):
    """Arguments for update_checklist_item tool."""

    name: str = Field(..., description="Checklist name that contains the item (without '.json').")
    item_id: str = Field(..., description="ID of the item to update (e.g., 'item_0').")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        ..., description="New status value to set for the item."
    )


@server.tool(
    name="create_checklist",
    description=(
        "Creates a new checklist with a given name and list of tasks. "
        "If checklist already exists, returns the existing one."
    ),
)
async def create_checklist(args: CreateChecklistArgs) -> str:
    # Check if checklist already exists
    existing_checklist = _load_checklist(args.name)
    if existing_checklist is not None:
        return (
            f"Checklist '{args.name}' already exists. "
            f"Returning existing checklist with {len(existing_checklist.items)} items.\n\n"
            + existing_checklist.model_dump_json(indent=2)
        )

    # Convert simple string items to ChecklistItem objects
    checklist_items = [
        ChecklistItem(id=f"item_{i}", description=item) for i, item in enumerate(args.items)
    ]
    checklist = Checklist(name=args.name, items=checklist_items)
    output_path = _save_checklist(checklist)
    return (
        f"Checklist '{args.name}' created with "
        f"{len(args.items)} items. Saved to {output_path}\n\n" + checklist.model_dump_json(indent=2)
    )


@server.tool(
    name="get_checklist",
    description="Retrieves the current state of a checklist by its name as a JSON string.",
)
async def get_checklist(args: GetChecklistArgs) -> str:
    checklist = _load_checklist(args.name)
    if checklist is None:
        return f"Checklist '{args.name}' not found."

    # Return the full checklist as a JSON string to provide all details, including IDs
    return checklist.model_dump_json(indent=2)


@server.tool(
    name="update_checklist_item",
    description="Updates the status of a specific item in a checklist.",
)
async def update_checklist_item(args: UpdateChecklistItemArgs) -> str:
    checklist = _load_checklist(args.name)
    if checklist is None:
        return f"Checklist '{args.name}' not found."

    # Find item index
    item_index = None
    for idx, item in enumerate(checklist.items):
        if item.id == args.item_id:
            item_index = idx
            break

    if item_index is None:
        return f"Item '{args.item_id}' not found in checklist '{args.name}'."

    # Validate sequential execution: can't start new item if previous ones aren't completed
    if args.status == "in_progress" and item_index > 0:
        for prev_idx in range(item_index):
            prev_item = checklist.items[prev_idx]
            if prev_item.status != "completed":
                return (
                    f"Cannot start item '{args.item_id}': "
                    f"Previous item '{prev_item.id}' ({prev_item.description}) "
                    f"is not completed (status: {prev_item.status}). "
                    f"Please complete previous items first."
                )

    # Update status
    checklist.items[item_index].status = args.status

    _save_checklist(checklist)
    return f"Updated item '{args.item_id}' in checklist '{args.name}' to status '{args.status}'."


def main() -> None:
    server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
