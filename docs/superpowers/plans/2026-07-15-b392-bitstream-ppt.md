# B392 Bitstream PowerPoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one concise, editable PowerPoint slide that explains how the b392 sender bitstream combines the base layer and tiered enhancement layer.

**Architecture:** Build a single 1280×720 slide with native PowerPoint shapes through `@oai/artifact-tool`. A proportional horizontal stream shows transmission order; a compact three-tier strip below explains patch counts and final bits per scalar.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, PowerPoint `.pptx`, bundled slide render and overflow-check tools.

## Global Constraints

- Use one 16:9 slide only.
- Keep the composition simple and presentation-ready.
- Use the terms “基础层”“增强层”“分档”; do not show Bucket, 分桶, or 粗桶.
- Show formulas as patch count × 768 scalars × allocated bits.
- Keep all text and shapes editable.
- Verify rendering, clipping, wrapping, overlap, and arithmetic before delivery.

---

### Task 1: Build and verify the bitstream slide

**Files:**
- Create: external scratch `tmp/create-b392-bitstream.mjs`
- Create: `outputs/b392_bitstream_structure.pptx`
- Test: rendered `slide-1.png`, slide layout JSON, and `slides_test.py` output in external scratch

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-15-bitstream-ppt-design.md`
- Produces: editable `outputs/b392_bitstream_structure.pptx`

- [ ] **Step 1: Initialize the artifact-tool workspace**

Run:

```powershell
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" --workspace "$TMP_DIR"
```

Expected: workspace initialized and `@oai/artifact-tool` resolvable from the scratch module.

- [ ] **Step 2: Author the one-slide presentation**

Create a plain `.mjs` module that uses these verified values:

```javascript
const values = {
  base: 196 * 768 * 1,       // 150528
  tier1Enh: 49 * 768 * 3,    // 112896
  tier2Enh: 49 * 768 * 1,    // 37632
  tier3Enh: 98 * 768 * 0,    // 0
};
const total = Object.values(values).reduce((sum, value) => sum + value, 0);
if (total !== 392 * 768) throw new Error(`Unexpected total: ${total}`);
```

The slide must contain:

```text
发送码流由基础层和分档增强层组成
基础层：196 patch × 768 标量 × 1 bit = 150,528 bit
第一档：49 patch × 768 标量 × 3 bit = 112,896 bit
第二档：49 patch × 768 标量 × 1 bit = 37,632 bit
第三档：98 patch × 768 标量 × 0 bit = 0 bit
总计：301,056 bit = 392 × 768 bit
最终量化精度：第一档 4 bit｜第二档 2 bit｜第三档 1 bit
```

- [ ] **Step 3: Export the PPTX and preview artifacts**

Run:

```powershell
node "$TMP_DIR/create-b392-bitstream.mjs"
```

Expected: `outputs/b392_bitstream_structure.pptx`, `slide-1.png`, and layout JSON are created.

- [ ] **Step 4: Verify structure and visual output**

Run:

```powershell
python "$SKILL_DIR/container_tools/slides_test.py" "outputs/b392_bitstream_structure.pptx"
python "$SKILL_DIR/container_tools/render_slides.py" "outputs/b392_bitstream_structure.pptx"
```

Expected: no slide-canvas overflow; rendered page has no clipped text, unwanted overlap, or broken Chinese characters.

- [ ] **Step 5: Deliver the verified deck**

Confirm the final file exists, has one slide, and the displayed arithmetic sums to `301,056 bit`.

### Task 2: Add the receiver parsing logic to the same slide

**Files:**
- Modify: external scratch `tmp/create-b392-bitstream.mjs`
- Create: `outputs/b392_bitstream_sender_receiver.pptx`
- Test: rendered slide PNG, layout JSON, and `slides_test.py` output in external scratch

**Interfaces:**
- Consumes: the sender bitstream layout and palette from Task 1
- Produces: one editable slide containing both sender structure and receiver parsing

- [ ] **Step 1: Compress the sender section into the upper half**

Keep the proportional base, first-tier enhancement, and second-tier enhancement segments. Preserve these formulas:

```text
196 patch × 768 标量 × 1 bit = 150,528 bit
49 patch × 768 标量 × 3 bit = 112,896 bit
49 patch × 768 标量 × 1 bit = 37,632 bit
98 patch × 768 标量 × 0 bit = 0 bit
```

- [ ] **Step 2: Add the four-step receiver flow**

Use native PowerPoint shapes and connectors for:

```text
解码基础层 → 基础重建结果 → 计算重要性并确定分档
增强层暂存 ─────────────→ 按分档读取增强比特
基础比特 + 增强比特 → 合并重建结果
```

Add one short conclusion:

```text
收端根据基础重建结果确定分档，无需额外接收分档元信息。
```

- [ ] **Step 3: Export a distinct PPTX**

Run the scratch module and save `outputs/b392_bitstream_sender_receiver.pptx`; preserve `outputs/b392_bitstream_structure.pptx`.

- [ ] **Step 4: Verify the edited slide**

Run `slides_test.py`, render the final slide, and inspect it at full size. Expected: one slide, no overflow, no unintended overlap, and all sender/receiver labels remain legible.
