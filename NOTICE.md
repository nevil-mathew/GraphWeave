# Third-Party Notice

GraphWeave began as a derivative of **tritopic**, a topic modeling library
published to PyPI by Roman Egger (SmartVisions AI) under the MIT License.
The original project's GitHub repository ([SmartVisions-AI/tritopic](https://github.com/SmartVisions-AI/tritopic))
did not include browsable source, so the starting codebase here was taken
from the published package ([`tritopic` on PyPI](https://pypi.org/project/tritopic/)).

Since then, GraphWeave has substantially rewritten and extended that starting
point — most core modules have roughly doubled or more in size, and several
subsystems were built from scratch and do not exist in the original project
at all:

- Cumulative / batch-wise streaming clustering (`graphweave.cumulative`)
- LLM-guided embedding adaptation via triplet fine-tuning (`graphweave.adaptation`)
- LLM-guided Leiden granularity calibration
- Quote verification for LLM-generated report narratives
- The adaptive FAISS/HNSW kNN backend
- Report-theme synthesis for qualitative research output

The MIT License requires that the original copyright and permission notice
be preserved in copies or substantial portions of the software. That notice
is reproduced below in full.

---

MIT License

Copyright (c) 2025 Roman Egger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
