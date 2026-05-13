# Annotation Guide — Syntactic Opening Classes

## 1. Overview

This guide supports the manual validation of an 8-class taxonomy of syntactic openings in Amazon reviews. The taxonomy was produced by an automated NLP pipeline (spaCy parsing → sentence-transformer embedding → KMeans clustering) and must now be checked against human judgment.

**Your task:** For each row in the annotation spreadsheet, read the 10-word opening of a review and assign it to one of 8 opening classes (labeled 0–7). Also rate your confidence in the assignment as high, medium, or low.

**Important rules:**

- Judge only the opening (the first 10 words). Do not consider what the full review might say.
- Focus on the syntactic structure — how the sentence begins — not the product or topic.
- Do not think about whether the review is helpful or unhelpful. Helpfulness is irrelevant to this task.
- If an opening could fit two classes, use the boundary rules in Section 3 below to decide.
- If you are genuinely unsure after consulting the boundary rules, assign your best guess and mark confidence as "low."

---

## 2. Class Definitions and Examples

### Class 0 — Product Reference

**Definition:** The opening points at the product using a demonstrative or article (this, the, a) followed by a product noun. The product itself is the grammatical subject or focus.

**Examples:**

- "This guide book covers a wide range of habitats: from"
- "I needed this cable 4-pin to 4-pin to get Hi-8"

---

### Class 1 — Acquisition Narrative

**Definition:** The opening tells a story about obtaining, purchasing, or first encountering the product. Past-tense acquisition verbs (bought, ordered, received, got, found) dominate.

**Examples:**

- "Bought this bag specifically to take to the Galapagos Islands"
- "I bought a copy of this years ago and remember"

---

### Class 2 — Evaluative Positioning

**Definition:** The opening uses complex syntactic machinery — concessions, conditionals, contrasts, hedges, or meta-commentary — to frame a stance before delivering judgment. Multiple clauses work together.

**Examples:**

- "I wasn't exactly sure why I purchased the most expensive"
- "Despite most people giving this lens a five-star rating, i"

---

### Class 3 — Direct Evaluation

**Definition:** The opening delivers an opinion or judgment immediately, with no preamble. The evaluative stance is the first thing expressed, using complete sentence structure.

**Examples:**

- "...you simply can't beat this deal! GREAT LITTLE TRIPOD! Does"
- "Like so many other fans, I've waited a grand total"

---

### Class 4 — Specific Identification

**Definition:** The opening is information-dense, anchoring the review to a specific product variant using model numbers, dates, proper nouns, or technical specifications.

**Examples:**

- "I purchased this DVD player in June 2000. In August,"
- "First of all, this product charges my Garmin 1450 very"

---

### Class 5 — Affective Stance

**Definition:** The opening leads with a first-person emotional state or attitude. The "I" + feeling/attitude verb combination is the signature (e.g., I love, I couldn't be happier, I was excited, I am not sure).

**Examples:**

- "I couldn't be happier with my Pentax 50mm lens and"
- "I was excited to begin my first Agatha Christie book"

---

### Class 6 — Contextual Framing

**Definition:** The opening provides factual background, scene-setting, or context before any evaluation. It states circumstances without using complex syntactic devices like concessions or conditionals.

**Examples:**

- "The only problem I have with this is that it"
- "This is a well made bag which can carry most"

---

### Class 7 — Terse Evaluation

**Definition:** The opening is fragmentary or compressed — short bursts, incomplete sentences, or non-standard syntax. Judgments are delivered in clipped, minimal form rather than full sentence structure.

**Examples:**

- "Junk, Only one ethernet port worked. Tech support sucked. Will"
- "Well edited and well brought out on high quality paper."
- "FOR PETE'S SAKE! THIS IS NOT A 3, 4 OR"

---

## 3. Boundary Rules

When an opening seems to fit two classes, use these rules to decide.

### Boundary 1 — Product Reference (0) vs Specific Identification (4)

Both refer to products. The difference is generic vs specific. If the opening includes model numbers, dates, or proper product names that narrow to a specific variant, code as **4**. If it uses generic demonstrative reference ("this book," "this cable") without identifying a specific variant, code as **0**.

### Boundary 2 — Product Reference (0) vs Acquisition Narrative (1)

An opening like "I bought this volume control" has both a purchase verb and a product reference. Look at what the sentence is about. If the main verb is about obtaining, buying, or receiving, code as **1**. If the product noun is the grammatical subject or focus, code as **0**.

### Boundary 3 — Acquisition Narrative (1) vs Affective Stance (5)

Both use first person. Look at what comes first structurally. If the emotional state verb leads ("I couldn't be happier," "I was excited"), code as **5**. If the acquisition or obtaining action leads ("I bought," "I ordered"), code as **1**.

### Boundary 4 — Direct Evaluation (3) vs Terse Evaluation (7)

Both deliver judgment. If the opening uses complete sentence structure with a subject and verb, code as **3**. If it is fragmentary, compressed, or lacks standard subject-verb form, code as **7**.

### Boundary 5 — Evaluative Positioning (2) vs Contextual Framing (6)

Both set up before evaluating. If the setup uses syntactic complexity — concessions ("despite," "although"), conditionals ("if"), hedges ("not sure why"), or contrasts — code as **2**. If it states factual context or background without those syntactic devices, code as **6**.

---

## 4. Annotation Procedure

1. Open `to_annotate.csv` in Excel or Google Sheets.
2. Hide the `cluster_label` column so you are not influenced by the pipeline's assignment.
3. Work through each row in order. For each opening:
   - Read the 10-word opening text.
   - Decide which of the 8 classes (0–7) best fits the syntactic pattern.
   - Enter the class number in the `manual_label` column.
   - Enter your confidence in the `confidence` column: high, medium, or low.
4. Do not skip rows. Every row must have a label and a confidence rating.
5. Work in focused blocks (e.g., 125 rows per sitting). Close notebooks and other materials while annotating — only this guide should be open.
6. Save your progress frequently.
7. When all 250 rows are complete, save the file as `annotated.csv` in `08_annotation/`.

---

## 5. Quick Reference Card

| ID | Class Name | Key Signal |
|----|-------------------------|---------------------------------------------|
| 0  | Product Reference       | Demonstrative/article + product noun        |
| 1  | Acquisition Narrative   | Past-tense acquisition verb                 |
| 2  | Evaluative Positioning  | Complex clauses: concessions, conditionals  |
| 3  | Direct Evaluation       | Opinion-first, full sentence                |
| 4  | Specific Identification | Model numbers, dates, proper nouns          |
| 5  | Affective Stance        | First-person + emotion/attitude verb        |
| 6  | Contextual Framing      | Factual background, no complex syntax       |
| 7  | Terse Evaluation        | Fragments, compressed, non-standard syntax  |

---

*Document created for Stage 5 — Manual Annotation Validation*
*Thesis: Syntactic Openings in Online Reviews*
*Tanishka Nath Pasumarthi | UMass Dartmouth | May 2026*
