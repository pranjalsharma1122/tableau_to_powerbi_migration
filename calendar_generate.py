import os
import json
import re
import sys
from typing import Dict, Any, Optional, Tuple, List

# Import Gemini API
try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google.generativeai library not found. Install with: pip install google-generativeai")
    sys.exit(1)

# ==================== CONSTANTS ====================
BASE_DIR = os.getcwd()
OUTPUT_DIR = r"C:\celendar\output"

# File paths
FINAL_JSON_PATH = os.path.join(OUTPUT_DIR, "final.json")
SCHEMA_JSON_PATH = os.path.join(OUTPUT_DIR, "schema_output.json")
POSITIONS_JSON_PATH = os.path.join(OUTPUT_DIR, "powerbi_chart_positions.json")
REFERENCE_TXT_PATH = os.path.join(OUTPUT_DIR, "Reference Chart Configurations.txt")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "visual_output.json")

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAwOgHgl1qu1wAEqteRGgwv80cCB_caDS4")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# ==================== HELPER FUNCTIONS ====================

def load_json_file(filepath: str) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load {filepath}: {e}")
        return None
    


def build_schema_mapping(schema_data: Dict) -> Dict[str, str]:
    """Build column to table mapping from schema_output.json."""
    mapping = {}
    for table_name, columns in schema_data.items():
        for col_info in columns:
            col_name = col_info.get("name", "").strip().lower()
            if col_name and col_name not in mapping:
                mapping[col_name] = table_name
    
    print(f"Built schema mapping with {len(mapping)} columns from {len(schema_data)} tables")
    return mapping


def find_calendar_chart_visual(final_data: list) -> Optional[Dict]:
    """Find the calendar chart visual in final.json."""
    for visual in final_data:
        chart_type = visual.get("chart_type", "").strip().lower()
        title = visual.get("title", "").strip().lower()
        source = visual.get("Source", "").strip().lower()
        
        if (chart_type == "calendarchart" or 
            "calendário" in title or 
            "calendario" in title or
            "calendar" in title or
            "calendário" in source or
            "calendar" in source):
            print(f"✓ Found calendar chart: '{visual.get('Source')}'")
            return visual
    
    print("ERROR: No calendar chart found")
    return None


def extract_calendar_prototype(reference_text: str) -> Optional[Dict]:
    """Extract calendar chart prototype from Reference Chart Configurations.txt."""
    try:
        reference_obj = json.loads(reference_text)
        config_str = reference_obj.get("config")
        
        if not config_str:
            print("ERROR: 'config' key not found in reference file")
            return None
        
        config_obj = json.loads(config_str)
        print("✓ Extracted calendar chart prototype")
        return config_obj
    
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to extract prototype: {e}")
        return None


# ==================== UNIVERSAL FIELD ROLE ENGINE ====================

class UniversalFieldRoleEngine:
    """
    TRULY UNIVERSAL field role detection engine.
    Works on ANY dataset in ANY language with ANY field names.
    """
    
    def __init__(self, schema_data: Dict, calendar_visual: Dict, schema_mapping: Dict):
        self.schema_data = schema_data
        self.calendar_visual = calendar_visual
        self.schema_mapping = schema_mapping
        self.all_fields = {}
        self.field_datatypes = {}
        
        self._extract_all_fields()
        self._get_field_datatypes()
        
    def _extract_all_fields(self):
        """Extract all fields from the calendar visual."""
        rows = self.calendar_visual.get("Rows") or {}
        columns = self.calendar_visual.get("Columns") or {}
        legend = self.calendar_visual.get("Legend") or {}
        
        self.all_fields = {**rows, **columns, **legend}
        
        print(f"\n{'='*70}")
        print(f"UNIVERSAL ENGINE: Found {len(self.all_fields)} fields in calendar visual")
        print(f"{'='*70}")
        for field_name, table in self.all_fields.items():
            print(f"  • {field_name} → {table}")
        print(f"{'='*70}\n")
    
    def _get_field_datatypes(self):
        """Get datatypes for all fields from schema."""
        for field_name, table_name in self.all_fields.items():
            if table_name not in self.schema_data:
                continue
            
            for col_info in self.schema_data[table_name]:
                if col_info.get("name", "").strip().lower() == field_name.strip().lower():
                    self.field_datatypes[field_name] = col_info.get("type", "").lower()
                    break
    
    def _ask_gemini_universal_classification(self) -> Dict[str, str]:
        """Use Gemini AI to classify fields UNIVERSALLY (works in any language/structure)."""
        try:
            # Prepare comprehensive context
            field_list = []
            for field_name, table in self.all_fields.items():
                datatype = self.field_datatypes.get(field_name, "unknown")
                field_list.append({
                    "name": field_name,
                    "table": table,
                    "datatype": datatype
                })
            
            hierarchy = self.calendar_visual.get("Hierarchy") or []
            aggregations_row = self.calendar_visual.get("Aggregation_row") or []
            
            prompt = f"""You are an expert Power BI data analyst. Analyze these fields from a calendar chart and classify each into EXACTLY ONE role.

FIELD CONTEXT:
Fields: {json.dumps(field_list, indent=2)}

HIERARCHY: {json.dumps(hierarchy, indent=2)}

AGGREGATIONS: {json.dumps(aggregations_row, indent=2)}

CALENDAR CHART REQUIREMENTS:
A Power BI calendar chart needs these 7 roles:
1. **date** - Primary date/datetime field (usually datetime type)
2. **event_index** - Unique identifier/row number (usually integer or string ID)
3. **week_label** - Label shown in calendar cells (numeric or categorical)
4. **category** - Grouping/classification field (usually string/text)
5. **legend** - Field for color coding (usually same as week_label or category)
6. **measure** - Numeric value field (integer/float)
7. **event_group** - Date hierarchy field for grouping (Year/Quarter/Month/Week)

CLASSIFICATION RULES:
- Date/datetime type → likely "date"
- Integer ID or "index"/"row" in name → likely "event_index"
- Numeric fields (int/float) → check if used in aggregations for "measure" or "week_label"
- String/text fields → likely "category"
- If hierarchy exists with date field → that's "event_group"
- "legend" often reuses "week_label" or "category"

IMPORTANT:
- Work with field names in ANY language (English, Portuguese, Spanish, etc.)
- Don't assume specific field names
- Use datatype and context to infer roles
- If a field appears in aggregations, it's likely "measure" or "week_label"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "date": "field_name",
  "event_index": "field_name",
  "week_label": "field_name",
  "category": "field_name",
  "legend": "field_name",
  "measure": "field_name",
  "event_group": "field_name"
}}"""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            gemini_roles = json.loads(response_text)
            
            print(f"\n{'='*70}")
            print("GEMINI UNIVERSAL CLASSIFICATION:")
            print(f"{'='*70}")
            for role, field in gemini_roles.items():
                datatype = self.field_datatypes.get(field, "unknown")
                print(f"  {role:15} → {field:30} (type: {datatype})")
            print(f"{'='*70}\n")
            
            return gemini_roles
        
        except Exception as e:
            print(f"WARNING: Gemini classification failed: {e}")
            return {}
    
    def _extract_hierarchy_info(self) -> Tuple[str, str]:
        """Extract hierarchy type and field from visual hierarchy - DYNAMIC VERSION."""
        hierarchy = self.calendar_visual.get("Hierarchy") or []
        
        # Default values
        hierarchy_type = "Month"
        hierarchy_field = None
        
        # Valid Power BI hierarchy types for calendar charts
        valid_types = ["Year", "Quarter", "Month", "Week", "Day"]
        
        print(f"\n{'='*70}")
        print("DYNAMIC HIERARCHY DETECTION")
        print(f"{'='*70}")
        print(f"Raw Hierarchy: {hierarchy}")
        
        for hier_str in hierarchy:
            # Extract hierarchy type (Year/Quarter/Month/Week/Day/Weekday)
            hier_match = re.search(r'(Year|Quarter|Month|Week|Weekday|Day)', hier_str, re.IGNORECASE)
            if hier_match:
                detected_type = hier_match.group(1).capitalize()
                
                # Power BI calendar charts don't support Weekday - convert to Week
                if detected_type == "Weekday":
                    print(f"  ⚠ Detected 'Weekday' → Converting to 'Week'")
                    detected_type = "Week"
                
                # Use the first valid hierarchy type found
                if detected_type in valid_types and hierarchy_type == "Month":
                    hierarchy_type = detected_type
                    print(f"  ✓ Detected hierarchy type: {hierarchy_type}")
            
            # Extract field name from hierarchy string
            # Format: "Quarter(TableName.FieldName)" or "Week(FieldName)"
            field_match = re.search(r'\((?:.*\.)?(.+?)\)', hier_str)
            if field_match:
                hierarchy_field = field_match.group(1).strip()
                print(f"  ✓ Detected hierarchy field: {hierarchy_field}")
        
        print(f"\nFinal: Type='{hierarchy_type}', Field='{hierarchy_field}'")
        print(f"{'='*70}\n")
        
        return hierarchy_type, hierarchy_field
    
    def _apply_intelligent_fallbacks(self, roles: Dict[str, str]) -> Dict[str, str]:
        """Apply intelligent fallback logic for missing/incorrect roles - MERGED VERSION."""
        available_fields = list(self.all_fields.keys())
        used_fields = set()
        
        # Get hierarchy info
        hierarchy_type, hierarchy_field = self._extract_hierarchy_info()
        
        # ========== DATE ROLE (HIGHEST PRIORITY) ==========
        if not roles.get("date"):
            # Try 1: Find datetime field
            for field in available_fields:
                datatype = self.field_datatypes.get(field, "")
                if any(dt in datatype for dt in ["date", "datetime", "timestamp"]):
                    roles["date"] = field
                    used_fields.add(field)
                    print(f"  → Fallback: date = {field} (datetime type)")
                    break
            
            # Try 2: Use hierarchy field
            if not roles.get("date") and hierarchy_field and hierarchy_field in available_fields:
                roles["date"] = hierarchy_field
                used_fields.add(hierarchy_field)
                print(f"  → Fallback: date = {hierarchy_field} (from hierarchy)")
            
            # Try 3: Use first available field
            if not roles.get("date") and available_fields:
                roles["date"] = available_fields[0]
                used_fields.add(available_fields[0])
                print(f"  → Fallback: date = {available_fields[0]} (first available)")
        else:
            used_fields.add(roles["date"])
        
        # ========== EVENT_GROUP ROLE ==========
        if not roles.get("event_group"):
            if hierarchy_field and hierarchy_field in available_fields:
                roles["event_group"] = hierarchy_field
                print(f"  → Fallback: event_group = {hierarchy_field} (from hierarchy)")
            elif roles.get("date"):
                roles["event_group"] = roles["date"]
                print(f"  → Fallback: event_group = {roles['date']} (reuse date)")
        
        # ========== MEASURE ROLE ==========
        if not roles.get("measure"):
            aggregations = self.calendar_visual.get("Aggregation_row") or []
            
            # Try 1: From aggregations
            for field in available_fields:
                if any(field in agg for agg in aggregations):
                    if field not in used_fields:
                        roles["measure"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: measure = {field} (from aggregations)")
                        break
            
            # Try 2: Numeric field
            if not roles.get("measure"):
                for field in available_fields:
                    datatype = self.field_datatypes.get(field, "")
                    if any(dt in datatype for dt in ["int", "float", "decimal", "number"]):
                        if field not in used_fields:
                            roles["measure"] = field
                            used_fields.add(field)
                            print(f"  → Fallback: measure = {field} (numeric type)")
                            break
            
            # Try 3: Any unused field
            if not roles.get("measure"):
                for field in available_fields:
                    if field not in used_fields:
                        roles["measure"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: measure = {field} (any available)")
                        break
        else:
            used_fields.add(roles["measure"])
        
        # ========== WEEK_LABEL ROLE ==========
        if not roles.get("week_label"):
            if roles.get("measure"):
                roles["week_label"] = roles["measure"]
                print(f"  → Fallback: week_label = {roles['measure']} (reuse measure)")
            else:
                for field in available_fields:
                    if field not in used_fields:
                        roles["week_label"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: week_label = {field}")
                        break
        
        # ========== EVENT_INDEX ROLE ==========
        if not roles.get("event_index"):
            # Try 1: ID/Index/Row/Col fields
            for field in available_fields:
                if any(pattern in field.lower() for pattern in ["id", "index", "row", "col"]):
                    if field not in used_fields:
                        roles["event_index"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: event_index = {field}")
                        break
            
            # Try 2: Any unused field
            if not roles.get("event_index"):
                for field in available_fields:
                    if field not in used_fields:
                        roles["event_index"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: event_index = {field}")
                        break
        else:
            used_fields.add(roles["event_index"])
        
        # ========== CATEGORY ROLE ==========
        if not roles.get("category"):
            # Try 1: String fields (not date field)
            for field in available_fields:
                if field == roles.get("date"):
                    continue
                datatype = self.field_datatypes.get(field, "")
                if any(dt in datatype for dt in ["text", "string", "varchar", "char"]):
                    if field not in used_fields:
                        roles["category"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: category = {field} (string type)")
                        break
            
            # Try 2: Any non-date unused field
            if not roles.get("category"):
                for field in available_fields:
                    if field != roles.get("date") and field not in used_fields:
                        roles["category"] = field
                        used_fields.add(field)
                        print(f"  → Fallback: category = {field}")
                        break
            
            # Try 3: Reuse any non-date field
            if not roles.get("category"):
                for field in available_fields:
                    if field != roles.get("date"):
                        roles["category"] = field
                        print(f"  → Fallback: category = {field} (reuse)")
                        break
        else:
            used_fields.add(roles["category"])
        
        # ========== LEGEND ROLE ==========
        if not roles.get("legend"):
            if roles.get("week_label"):
                roles["legend"] = roles["week_label"]
                print(f"  → Fallback: legend = {roles['week_label']} (reuse week_label)")
            elif roles.get("category"):
                roles["legend"] = roles["category"]
                print(f"  → Fallback: legend = {roles['category']} (reuse category)")
        
        return roles
    
    def detect_roles(self) -> Dict[str, Tuple[str, str, str]]:
        """
        Main method: Detect all field roles using universal logic.
        Returns: Dict mapping role to (queryRef, table, hierarchy_type)
        """
        print(f"\n{'='*70}")
        print("UNIVERSAL FIELD ROLE DETECTION - STARTING")
        print(f"{'='*70}\n")
        
        # Step 1: Gemini AI classification
        print("[1/3] Running Gemini AI universal classification...")
        gemini_roles = self._ask_gemini_universal_classification()
        
        # Step 2: Apply intelligent fallbacks
        print("[2/3] Applying intelligent fallback logic...")
        final_roles = self._apply_intelligent_fallbacks(gemini_roles)
        
        # Step 3: Build queryRef format
        print("[3/3] Building Power BI queryRef format...")
        
        hierarchy_type, _ = self._extract_hierarchy_info()
        
        result = {}
        for role, field_name in final_roles.items():
            if not field_name or field_name not in self.all_fields:
                print(f"  ⚠ Warning: Role '{role}' has invalid field '{field_name}'")
                continue
            
            table_name = self.all_fields[field_name]
            
            # Build queryRef - CORRECT DYNAMIC FORMAT
            if role == "event_group":
                # Format: Table.Field.Variation.Date Hierarchy.Type
                queryref = f"{table_name}.{field_name}.Variation.Date Hierarchy.{hierarchy_type}"
            else:
                queryref = f"{table_name}.{field_name}"
            
            result[role] = (queryref, table_name, hierarchy_type)
            print(f"  ✓ {role:15} → {queryref}")
        
        print(f"\n{'='*70}")
        print("FIELD ROLE DETECTION COMPLETE")
        print(f"{'='*70}\n")
        
        return result


# ==================== CONFIGURATION UPDATE FUNCTIONS ====================

def find_chart_position(chart_title: str, positions_data: list) -> Dict[str, float]:
    """Find position data for chart by title."""
    title_lower = chart_title.strip().lower()
    
    for pos_entry in positions_data:
        entry_title = pos_entry.get("chart", "").strip().lower()
        if (entry_title == title_lower or 
            "calendário" in entry_title or 
            "calendario" in entry_title or 
            "calendar" in entry_title):
            return {
                "x": float(pos_entry.get("x", 0)),
                "y": float(pos_entry.get("y", 0)),
                "z": float(pos_entry.get("z", 0)),
                "width": float(pos_entry.get("width", 1100)),
                "height": float(pos_entry.get("height", 400))
            }
    
    print(f"  ⚠ No position found for '{chart_title}', using defaults")
    return {"x": 17.05, "y": 290.69, "z": 0.0, "width": 1103.83, "height": 413.16}


def update_calendar_config(prototype: Dict, field_mappings: Dict, position: Dict) -> Dict:
    """Update the prototype with new field mappings and position."""
    import copy
    config = copy.deepcopy(prototype)
    
    # Extract values from field_mappings
    events_ref, events_table, _ = field_mappings['event_index']
    group_ref, group_table, hierarchy_type = field_mappings['event_group']
    color_ref, color_table, _ = field_mappings['week_label']
    date_ref, date_table, _ = field_mappings['date']
    
    # Update projections
    if "singleVisual" in config and "projections" in config["singleVisual"]:
        projections = config["singleVisual"]["projections"]
        
        projections["events"] = [{"queryRef": events_ref}]
        projections["EventGroup"] = [{"queryRef": group_ref}]
        projections["CellColor"] = [{"queryRef": color_ref}]
        projections["StartDate"] = [{"queryRef": date_ref}]
        projections["EndDate"] = [{"queryRef": date_ref}]
    
    # Update prototypeQuery
    if "singleVisual" in config and "prototypeQuery" in config["singleVisual"]:
        proto_query = config["singleVisual"]["prototypeQuery"]
        
        # Update From clause
        if "From" in proto_query:
            proto_query["From"] = [{"Name": "s", "Entity": events_table, "Type": 0}]
        
        # Update Select clause - FINAL CORRECT VERSION
        if "Select" in proto_query:
            event_field = events_ref.split('.')[-1]
            date_field = date_ref.split('.')[-1]
            color_field = color_ref.split('.')[-1]
            
            proto_query["Select"] = [
                {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": "s"}},
                        "Property": event_field
                    },
                    "Name": events_ref,
                    "NativeReferenceName": event_field
                },
                {
                    "HierarchyLevel": {
                        "Expression": {
                            "Hierarchy": {
                                "Expression": {
                                    "PropertyVariationSource": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Name": "Variation",
                                        "Property": date_field
                                    }
                                },
                                "Hierarchy": "Date Hierarchy"
                            }
                        },
                        "Level": hierarchy_type
                    },
                    "Name": group_ref,
                    "NativeReferenceName": f"{date_field} {hierarchy_type}"
                },
                {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": "s"}},
                        "Property": color_field
                    },
                    "Name": color_ref,
                    "NativeReferenceName": color_field
                },
                {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": "s"}},
                        "Property": date_field
                    },
                    "Name": date_ref,
                    "NativeReferenceName": date_field
                }
            ]
    
    # Update position
    if "layouts" in config and len(config["layouts"]) > 0:
        config["layouts"][0]["position"]["x"] = position["x"]
        config["layouts"][0]["position"]["y"] = position["y"]
        config["layouts"][0]["position"]["z"] = position["z"]
        config["layouts"][0]["position"]["width"] = position["width"]
        config["layouts"][0]["position"]["height"] = position["height"]
    
    return config


def validate_calendar_config(config_obj: Dict) -> bool:
    """Validate the calendar configuration structure."""
    try:
        if "singleVisual" not in config_obj:
            print("  ✗ Validation failed: Missing singleVisual")
            return False
        
        projections = config_obj.get("singleVisual", {}).get("projections", {})
        required_projections = ["events", "EventGroup", "CellColor", "StartDate", "EndDate"]
        
        for proj in required_projections:
            if proj not in projections:
                print(f"  ✗ Validation failed: Missing {proj} projection")
                return False
            
            if not isinstance(projections[proj], list) or len(projections[proj]) == 0:
                print(f"  ✗ Validation failed: {proj} projection is empty")
                return False
        
        print("  ✓ Configuration validation passed")
        return True
    
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False


# ==================== MAIN GENERATION FUNCTION ====================

def generate_calendar_chart():
    """Main function to generate the calendar chart configuration."""
    print("=" * 70)
    print("Universal Calendar Chart Generator v3.0 - MERGED")
    print("Works on ANY dataset in ANY language!")
    print("=" * 70)
    
    # Load input files
    print("\n[1/7] Loading input files...")
    
    final_data = load_json_file(FINAL_JSON_PATH)
    schema_data = load_json_file(SCHEMA_JSON_PATH)
    positions_data = load_json_file(POSITIONS_JSON_PATH)
    
    if not all([final_data, schema_data, positions_data]):
        print("ERROR: Critical files missing. Aborting.")
        return 1
    
    # Load reference text
    try:
        with open(REFERENCE_TXT_PATH, 'r', encoding='utf-8') as f:
            reference_text = f.read()
        print("  ✓ Loaded reference text")
    except Exception as e:
        print(f"  ✗ ERROR: Could not load reference text: {e}")
        return 1
    
    # Find calendar chart visual
    print("\n[2/7] Finding calendar chart visual...")
    calendar_visual = find_calendar_chart_visual(final_data)
    if not calendar_visual:
        print("ERROR: No calendar chart found. Aborting.")
        return 1
    
    # Build schema mapping
    print("\n[3/7] Building schema mapping...")
    schema_mapping = build_schema_mapping(schema_data)
    
    # Find chart position
    print("\n[4/7] Finding chart position...")
    chart_title = calendar_visual.get("title", "") or calendar_visual.get("Source", "")
    position = find_chart_position(chart_title, positions_data)
    print(f"  ✓ Position: x={position['x']}, y={position['y']}, w={position['width']}, h={position['height']}")
    
    # Extract calendar chart prototype
    print("\n[5/7] Extracting calendar chart prototype...")
    prototype = extract_calendar_prototype(reference_text)
    if not prototype:
        print("ERROR: Failed to extract prototype. Aborting.")
        return 1
    
    # UNIVERSAL DYNAMIC FIELD DETECTION
    print("\n[6/7] Running universal field detection engine...")
    engine = UniversalFieldRoleEngine(schema_data, calendar_visual, schema_mapping)
    field_mappings = engine.detect_roles()
    
    if len(field_mappings) < 7:
        print(f"WARNING: Only {len(field_mappings)} roles detected (expected 7)")
    
    # Update prototype with new fields and position
    print("\n[7/7] Updating configuration...")
    final_config = update_calendar_config(prototype, field_mappings, position)
    
    # Final validation
    if not validate_calendar_config(final_config):
        print("ERROR: Final configuration is invalid")
        return 1
    
    # Create output structure
    output_visual = {
        "config": json.dumps(final_config, ensure_ascii=False),
        "filters": "[]",
        "x": position["x"],
        "y": position["y"],
        "z": position["z"],
        "width": position["width"],
        "height": position["height"]
    }
    
    # Save to file
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_visual, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully saved output to: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save output: {e}")
        return 1
    
    # Success message
    print("\n" + "=" * 70)
    print("=" * 70)
    print(f"\nOutput file: {OUTPUT_JSON_PATH}")
    print("=" * 70)
    
    return 0


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    try:
        exit_code = generate_calendar_chart()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)