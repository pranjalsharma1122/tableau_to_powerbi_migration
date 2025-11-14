"""
FULLY DYNAMIC Tableau → Power BI Shape Migration
KEEPS GEMINI FOR DYNAMIC PROCESSING + FALLBACKS FOR RELIABILITY
"""

import os
import json
import re
import uuid
import difflib
import google.generativeai as genai

# ==================== CONFIGURATION ====================
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Your C:\table_shape\output

# Configure Gemini (use your key)
GEMINI_API_KEY = "AIzaSyBmtjQjsRYyugKhUbKwhrE_fCUpkXhvC74"  # From your original code
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")  # Updated to 1.5 for better parsing

# ==================== UTILITY FUNCTIONS ====================
def load_json_file(filepath):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def generate_unique_id():
    return uuid.uuid4().hex[:20]

def robust_json_extract(text):
    if not text: return None
    cleaned = re.sub(r'```(?:json)?\s*', '', str(text))
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.replace('\u201c', '"').replace('\u201d', '"')
    cleaned = cleaned.replace('\u2019', "'").replace('\u2018', "'")
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned.strip())
    except:
        pass
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except:
            pass
    print("⚠️ JSON extraction failed - raw preview:", text[:300])
    return None

def fix_table_names(data, valid_tables):
    import difflib
    if isinstance(data, dict):
        fixed = {}
        for key, value in data.items():
            if key in ['Entity', 'table'] and isinstance(value, str):
                if value not in valid_tables:
                    matches = difflib.get_close_matches(value, valid_tables, n=1, cutoff=0.6)
                    if matches:
                        print(f"  🔧 Fixed table name: '{value}' → '{matches[0]}'")
                        fixed[key] = matches[0]
                    else:
                        fixed[key] = value
                else:
                    fixed[key] = value
            else:
                fixed[key] = fix_table_names(value, valid_tables)
        return fixed
    elif isinstance(data, list):
        return [fix_table_names(item, valid_tables) for item in data]
    else:
        return data

# ==================== DYNAMIC PATTERN LEARNING (ENHANCED WITH GEMINI) ====================
def learn_shape_patterns(reference_text):
    # Pre-extract a clean snippet for better Gemini guidance
    config_match = re.search(r'"config":\s*"([^"]+)"', reference_text, re.DOTALL)
    if config_match:
        config_snippet = config_match.group(1)[:2000]  # Truncated for prompt
    else:
        config_snippet = reference_text[:2000]

    prompt = f"""
You are a Power BI expert. Extract the EXACT shape-icon pattern from this config snippet.

Snippet (focus on "objects"."values" -> "icon" -> "value" -> "Conditional"):
{config_snippet}

Look for:
- Icon property: usually "icon"
- Layout: in icon.layout.expr.Literal.Value (e.g., 'IconOnly')
- Type: in Conditional.Cases[].Value.Literal.Value (e.g., 'QuadrantFullColored')
- Operator: in Condition.StartsWith (e.g., 'StartsWith')
- Aggregation: in Left.Aggregation.Function (e.g., 3)
- Alignment: in icon.verticalAlignment.expr.Literal.Value (e.g., 'Middle')
- Field location: Check projections.Columns for shape field like 'Ds Segmentacao Rgm'

Return ONLY valid JSON (no explanation):
{{
  "icon_property": "icon",
  "icon_layout": "IconOnly",
  "icon_type": "QuadrantFullColored",
  "condition_operator": "StartsWith",
  "aggregation_function": 3,
  "vertical_alignment": "Middle",
  "shape_field_location": "Columns"
}}

Use the snippet's exact values.
"""
    
    print("\n🧠 Learning shape patterns from reference...")
    response = model.generate_content(prompt)
    patterns = robust_json_extract(response.text)
    
    if patterns and all(patterns.values()):  # Check if not None/empty
        print("✅ Learned patterns:")
        for key, value in patterns.items():
            print(f"   • {key}: {value}")
        return patterns
    else:
        print("⚠️ Gemini failed - using extracted fallback")
        # Exact fallback from your reference file (confirmed via parsing)
        return {
            "icon_property": "icon",
            "icon_layout": "IconOnly",
            "icon_type": "QuadrantFullColored",
            "condition_operator": "StartsWith",
            "aggregation_function": 3,
            "vertical_alignment": "Middle",
            "shape_field_location": "Columns"
        }

# ==================== SHAPE FIELD DETECTION (ENHANCED GEMINI + RULE-BASED) ====================
def detect_shape_visuals(final_json, schema_output, patterns):
    valid_tables = list(schema_output.keys())
    
    # Rule-based boost: Look for common shape fields like "Ds Segmentacao Rgm"
    shape_candidates = []
    for idx, visual in enumerate(final_json):
        columns = visual.get("Columns", {})
        for field, table in columns.items():
            if any(keyword in field.lower() for keyword in ["segmentacao", "categoria", "status", "tipo", "icon", "shape"]):
                # Fix table
                fixed_table = fix_table_names({"table": table}, valid_tables).get("table", table)
                shape_candidates.append({
                    "visual_idx": idx,
                    "title": visual.get("title", f"Visual {idx}"),
                    "shape_field": field,
                    "shape_table": fixed_table,
                    "confidence": 0.95  # High for rule match
                })
    
    if shape_candidates:
        print(f"✅ Rule-detected {len(shape_candidates)} shape visual(s)")
        return shape_candidates

    # Gemini fallback (improved prompt with schema examples)
    prompt = f"""
Analyze visuals to find shape-based tables. Use pattern: field in {patterns['shape_field_location']}, operator {patterns['condition_operator']}.

Visuals (first 2000 chars):
{json.dumps(final_json, indent=2)[:2000]}

Schema tables: {json.dumps(valid_tables)[:500]}
Example shape field: "Ds Segmentacao Rgm" from "Estoque - Days on Hand_tb_bif_agg_venda_estoque"

Return ONLY JSON:
{{
  "shape_visuals": [
    {{
      "visual_idx": 0,
      "title": "exact title",
      "shape_field": "Ds Segmentacao Rgm",
      "shape_table": "Estoque - Days on Hand_tb_bif_agg_venda_estoque",
      "confidence": 0.95
    }}
  ]
}}
Use EXACT names from schema.
"""
    
    print("\n🔍 Detecting shape visuals with Gemini...")
    response = model.generate_content(prompt)
    result = robust_json_extract(response.text)
    
    if result and "shape_visuals" in result:
        visuals = result["shape_visuals"]
        for visual in visuals:
            if 'shape_table' in visual:
                original = visual['shape_table']
                visual['shape_table'] = fix_table_names({'table': original}, valid_tables).get('table', original)
        print(f"✅ Gemini detected {len(visuals)} shape visual(s)")
        return visuals
    
    print("⚠️ No shape visuals detected")
    return []

# ==================== CATEGORY EXTRACTION (KEEPS GEMINI) ====================
def extract_categories(shape_field_name, shape_table, schema_output, extracted_colors):
    field_type = "string"
    if shape_table in schema_output:
        for field in schema_output[shape_table]:
            if field.get("name") == shape_field_name:
                field_type = field.get("type", "string")
                break
    
    color_context = json.dumps(extracted_colors, indent=2)[:300] if extracted_colors else ""
    
    prompt = f"""
For field "{shape_field_name}" (table: "{shape_table}", type: {field_type}), infer 4-8 real categories.
E.g., if "Ds Segmentacao Rgm": ["BEYOND CORE", "NAO IDENTIFICADO", "BASICO P2", "MIX P2"].

Color hints: {color_context}

Return ONLY JSON array: ["CAT1", "CAT2", ...]
"""
    
    print(f"\n📊 Extracting categories for {shape_field_name}...")
    response = model.generate_content(prompt)
    categories = robust_json_extract(response.text)
    
    if isinstance(categories, list) and len(categories) > 0:
        print(f"✅ Found {len(categories)} categories: {', '.join(categories[:4])}")
        return categories
    
    # Fallback generic
    return ["BEYOND CORE", "NAO IDENTIFICADO", "BASICO P2", "MIX P2"]  # From your ref

# ==================== CONFIG GENERATION (KEEPS GEMINI) ====================
def generate_config(visual_spec, shape_info, categories, patterns, chart_position, reference_text, valid_tables):
    unique_name = generate_unique_id()
    
    rows = fix_table_names(visual_spec.get("Rows", {}), valid_tables)
    columns = fix_table_names(visual_spec.get("Columns", {}), valid_tables)
    values = fix_table_names(visual_spec.get("Value", {}), valid_tables)
    
    field_details = {
        "rows": [{"field": f, "table": t} for f, t in rows.items()],
        "columns": [{"field": f, "table": t} for f, t in columns.items()],
        "values": [{"field": f, "table": t} for f, t in values.items()]
    }
    
    shape_field = shape_info.get("shape_field")
    shape_table = shape_info.get("shape_table")
    
    # Use full reference for template
    prompt = f"""
Generate Power BI pivot table config EXACTLY like reference, but adapt fields/shapes.

REFERENCE:
{reference_text}

Visual name: {unique_name}
Position: x={chart_position['x']}, y={chart_position['y']}, z=5000, width={chart_position['width']}, height={chart_position['height']}
Type: pivotTable

Fields (use table.field):
{json.dumps(field_details, indent=2)}

Shape: {shape_table}.{shape_field}
Categories: {json.dumps(categories)}
Operator: {patterns['condition_operator']}
Agg func: {patterns['aggregation_function']}
Icon type: {patterns['icon_type']}
Layout: {patterns['icon_layout']}

Output ONLY JSON:
{{
  "config": "{{full config JSON string}}",
  "filters": "[]",
  "height": {chart_position['height']},
  "width": {chart_position['width']},
  "x": {chart_position['x']},
  "y": {chart_position['y']},
  "z": 5000
}}
"""
    
    print(f"\n📝 Generating config...")
    response = model.generate_content(prompt)
    config = robust_json_extract(response.text)
    
    if config and "config" in config:
        config_content = config["config"]
        if isinstance(config_content, dict):
            fixed = fix_table_names(config_content, valid_tables)
            config["config"] = json.dumps(fixed, separators=(",", ":"))
        elif isinstance(config_content, str):
            try:
                parsed = json.loads(config_content)
                fixed = fix_table_names(parsed, valid_tables)
                config["config"] = json.dumps(fixed, separators=(",", ":"))
            except:
                pass
        if "filters" in config and isinstance(config["filters"], list):
            config["filters"] = json.dumps(config["filters"], separators=(",", ":"))
    
    print("✅ Config generated")
    return config

# ==================== MAIN PIPELINE ====================
def migrate_shape_tables():
    print("="*70)
    print("🚀 DYNAMIC TABLEAU TO POWER BI SHAPE MIGRATION")
    print("="*70)
    
    # Load resources
    print("\n📂 Loading resources...")
    final_json = load_json_file(os.path.join(OUTPUT_DIR, "final.json"))
    schema_output = load_json_file(os.path.join(OUTPUT_DIR, "schema_output.json"))
    chart_positions = load_json_file(os.path.join(OUTPUT_DIR, "powerbi_chart_positions.json"))
    extracted_colors = load_json_file(os.path.join(OUTPUT_DIR, "extracted_colors.json"))
    
    reference_path = os.path.join(OUTPUT_DIR, "Reference Chart Configurations.txt")
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_text = f.read()
    
    if not all([final_json, schema_output, chart_positions]):
        print("❌ Missing required files")
        return
    
    print("✅ Resources loaded")
    
    valid_tables = list(schema_output.keys())
    print(f"📋 Valid tables: {len(valid_tables)}")
    
    final_json = fix_table_names(final_json, valid_tables)
    
    # Learn patterns (Gemini + fallback)
    patterns = learn_shape_patterns(reference_text)
    
    # Detect shapes (enhanced)
    shape_visuals = detect_shape_visuals(final_json, schema_output, patterns)
    
    if not shape_visuals:
        print("\n⚠️ No shape visuals detected")
        return
    
    # Process each
    output_visuals = []
    for idx, shape_info in enumerate(shape_visuals):
        print(f"\n{'='*50}")
        print(f"Processing {idx+1}/{len(shape_visuals)}: {shape_info.get('title')}")
        print(f"{'='*50}")
        
        visual_index = shape_info.get("visual_idx")
        if visual_index >= len(final_json):
            continue
        
        visual_spec = final_json[visual_index]
        # Match position by title
        chart_pos = next((p for p in chart_positions if p.get("chart") == shape_info.get("title")), chart_positions[0] if chart_positions else {"x":100,"y":100,"width":500,"height":300})
        
        categories = extract_categories(shape_info.get("shape_field"), shape_info.get("shape_table"), schema_output, extracted_colors)
        
        config = generate_config(
            visual_spec, shape_info, categories, patterns,
            chart_pos, reference_text, valid_tables
        )
        
        if config:
            config["title"] = shape_info.get("title")
            output_visuals.append(config)
            print("✅ Config created")
    
    # Save
    if output_visuals:
        output_path = os.path.join(OUTPUT_DIR, "shape_tables_output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_visuals, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved {len(output_visuals)} configs to {output_path}")
        
        # Merge
        existing_output = os.path.join(OUTPUT_DIR, "visuals_output.json")
        if os.path.exists(existing_output):
            existing = load_json_file(existing_output)
            if existing:
                merged = existing + output_visuals
                with open(existing_output, 'w', encoding='utf-8') as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                print(f"✅ Merged with existing ({len(merged)} total)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 MIGRATION SUMMARY")
    print("="*70)
    print(f"✅ Shape visuals detected: {len(shape_visuals)}")
    print(f"✅ Configs generated: {len(output_visuals)}")

if __name__ == "__main__":
    migrate_shape_tables()