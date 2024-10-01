
# CDR Analysis Dashboard

This project provides a Call Detail Records (CDR) Analysis Dashboard built using Streamlit.

## Setup Instructions

### 1. Create a Python Virtual Environment
It's recommended to create a virtual environment to manage dependencies. You can create a virtual environment using the following command:

```bash
python -m venv env
```

### 2. Activate the Virtual Environment

- **Windows**:
  ```bash
  .\env\Scripts\activate
  ```
- **Linux/Mac**:
  ```bash
  source env/bin/activate
  ```

### 3. Install the Required Dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Navigate to the Dashboard Directory

```bash
cd dashboard
```

### 5. Run the Streamlit App

Once inside the `dashboard` directory, run the Streamlit application:

```bash
streamlit run "CDR Dashboard.py"
```

### 6. Access the Dashboard

After running the above command, Streamlit will provide a local URL. Open the link in your web browser to access the CDR Analysis Dashboard.

---

### Additional Notes:

- Ensure you have Streamlit installed (`streamlit` is included in `requirements.txt`).
- The dashboard is designed to analyze CDR data and visualize results interactively.
