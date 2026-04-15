# Usage

- Launch the Streamlit UI:
  ```bash
  streamlit run ui/app.py
  ```
- Enter a blog topic and start generation.
- The system executes the LangGraph pipeline through `graph/workflow.py` and writes outputs to `outputs/`.
- Generated assets appear in `outputs/images/`, `outputs/blogs/`, and `outputs/logs/`.
- Run tests with:
  ```bash
  pytest tests/
  ```
