# 04 · Aviation Regulations RAG Assistant

## Problem
Aviation regulations are long, cross-referenced and hard to search. Operators, including drone operators, need fast answers they can trust, with the exact source cited.

## Data
Public regulatory documents: Nigerian Civil Aviation Regulations (drone/RPAS sections) and FAA handbooks.

## Approach
1. Parse the PDFs and chunk them by section.
2. Embed the chunks and store them in a local vector index.
3. Retrieve relevant sections and generate an answer with citations using an LLM.
4. Evaluate: a hand-built question set scored for answer correctness and citation accuracy.

## Results
_To be added._

## Run
_To be added._
