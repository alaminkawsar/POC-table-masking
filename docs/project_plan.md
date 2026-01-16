# User-Defined Sensitive Information Detection and Masking in UI Images

## 📌 Project Overview
This research project focuses on automatically detecting and masking sensitive information in **UI screenshots** (web applications) based on **user-defined rules**. The system analyzes UI elements such as tables, text fields, dropdowns, and labels, extracts text, understands layout structure, and applies visual masking to specified values.

---

## 🎯 Objectives

### Primary Objectives
- Detect UI elements in screenshots
- Extract text with spatial awareness
- Understand layout structures (tables, fields)
- Apply user-defined masking rules
- Output visually masked UI images

### Secondary Objectives
- Minimize false positives
- Preserve UI readability
- Support flexible masking strategies

---

## 🧩 Problem Statement
Given a UI image containing multiple UI elements, the system should:
1. Detect UI components
2. Extract text from them
3. Identify sensitive content based on user rules
4. Mask those values in the image

**For example:** Masing table header: **Rev Nbr**, Table column values: **Line Number**, Textfield value with label: **Order Customer**, **Sold To**, **Ship To**.
- **Input Image** <br> <img src="input.jpeg" alt="Input Image Example" width="400"/>
- **Output Image** <br> <img src="output.jpg" alt="Output Image Example" width="400"/>
<br><br>
---

## 🔍 Challenges
- Diverse UI layouts and styles
- OCR inaccuracies
- Ambiguous layout structures
- Dynamic user-defined masking requirements

---

## 🧠 Research Questions
1. How accurately can UI elements be detected?
2. How reliable is text-to-layout mapping?
3. How flexible and effective are user-defined masking rules?
4. What is the tradeoff between accuracy and performance?

---

## 📚 Literature Review
Relevant research areas:
- Document Layout Analysis
- UI Element Detection
- OCR Systems
- Information Redaction

Key references:
- PubLayNet
- DocLayNet
- LayoutLM / LayoutLMv3
- YOLO / Detectron2
- PaddleOCR / Tesseract

**Identified Gap:**  
Existing solutions focus on fixed PII masking and document layouts, lacking dynamic, user-defined masking in UI screenshots.

---

## 🏗️ System Architecture
```
Input UI Image
↓
UI Element Detection
↓
Text Detection & OCR
↓
Layout & Structure Mapping
↓
User Rule Engine
↓
Masking Engine
↓
Masked Output Image
```


---

## 🗂️ Dataset Strategy

### Data Sources
- Mobile & Web app screenshots
- Synthetic UI images
- Public layout datasets

### Annotation
- UI element type
- Bounding boxes
- Table structure
- Text content

Tools:
- CVAT
- LabelImg
- Roboflow

---

## 🔬 Methodology

### UI Element Detection
- Model: YOLOv8 / Detectron2
- Classes: Table, Text Field, Dropdown, Radio Button, Label

### Text Detection & OCR
- PaddleOCR (primary)
- Tesseract (baseline)

### Layout Understanding
- Table grid detection
- Text-to-cell mapping
- Spatial heuristics

Optional:
- LayoutLM for spatial reasoning

---

## 🧾 User-Defined Masking Rules

Supported Rule Types:
- Exact text match
- Regex-based masking
- Column-based masking
- Element-type based masking
- Position-based masking

Rule Execution Flow:
Rule → Match Text → Match Location → Mask Decision


---

## 🎨 Masking Strategies
- Black rectangle
- Blur
- Pixelation
- Replace with "***"

Comparison Criteria:
- Privacy strength
- Readability
- False masking rate

---

## 🧪 Prototype (PoC)

### Goals
- Detect tables and text
- Mask table columns by header
- Mask specific text patterns

### Tech Stack
- Python
- OpenCV
- PaddleOCR
- YOLOv8
- NumPy

---

## 📊 Evaluation Metrics

### Detection Metrics
- Precision / Recall
- IoU

### Masking Metrics
- Mask Precision
- Mask Recall
- False Mask Rate

### UX Metrics
- Visual clarity score
- Rule success rate

---

## 🧪 Experiments

| Experiment | Description |
|----------|------------|
| Baseline | OCR-only masking |
| Exp 1 | OCR + table detection |
| Exp 2 | OCR + layout reasoning |
| Exp 3 | Rule complexity stress test |

---

## 🚀 Productization Plan

### API Input Example
```json
{
  "image": "...",
  "rules": [
    { "type": "column", "name": "Ship To" },
    { "type": "regex", "pattern": "\\\\d{11}" }
  ]
}

