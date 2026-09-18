<video width="100%" autoplay loop muted playsinline>
  <source src="docs/Optim.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# Optim.AI - Intelligent B2B Inventory Management & Decision Support

## 📊 Overview

**Optim.AI** is a powerful Streamlit-based web application that helps B2B businesses make data-driven decisions by automatically extracting, analyzing, and comparing quotation emails from multiple suppliers. It simplifies the procurement process by aggregating key information like delivery dates, brand names, and prices, while also providing real-time brand sentiment analysis.

---

## 🎯 Who Is This Project For?

Optim.AI is designed for:

- **B2B Procurement Teams** - Professionals who manage supplier relationships and purchase inventory
- **Supply Chain Managers** - Individuals responsible for sourcing products from multiple vendors
- **Small to Medium Business Owners** - Entrepreneurs who need to compare supplier quotes efficiently
- **Inventory Managers** - Staff tracking delivery schedules and pricing across suppliers

---

## 🔍 What Problem Does It Solve?

### The Challenge

B2B businesses often receive numerous quotation emails from different suppliers for the same products. Manually:

1. **Reading through dozens of emails** to find delivery dates, prices, and brand information
2. **Comparing prices** across multiple suppliers to find the best deal
3. **Checking delivery timelines** to ensure stock availability
4. **Researching brand reputations** before making purchasing decisions

This process is **time-consuming, error-prone, and inefficient**.

### The Solution

Optim.AI automates the entire quote comparison process by:

- ✅ Extracting **delivery dates**, **quoted prices**, and **brand names** from quotation emails using AI
- ✅ Sorting and presenting the **earliest delivery option** and **cheapest option**
- ✅ Fetching **real-time brand reviews** from social media (Twitter)
- ✅ Performing **sentiment analysis** on brand reviews to help make informed decisions

---

## ⚙️ How Does It Work?

### Step 1: Authentication
- User enters their **Gmail credentials** and **App Password** (for secure IMAP access)
- User provides their **OpenAI API Key** for AI-powered data extraction

### Step 2: Email Search
- User enters a **keyword** to search for relevant quotation emails in their inbox
- The app connects to Gmail via IMAP and fetches all matching emails

### Step 3: Data Extraction
- For each email, the app extracts:
  - **Sender information**
  - **Subject line**
  - **Email body content**
- **OpenAI's GPT model** (text-davinci-003) intelligently parses the email body to extract:
  - 📅 **Delivery Date** (in YYYY-MM-DD format)
  - 💰 **Quoted Price**
  - 🏷️ **Brand Name**

### Step 4: Analysis & Comparison
- All extracted data is organized into a **sortable DataFrame**
- The app identifies and displays:
  - **Earliest Possible Option** - Supplier with the fastest delivery
  - **Cheapest Option** - Supplier with the lowest price

### Step 5: Brand Sentiment Analysis
- User selects a brand from the extracted data
- The app fetches **real-time reviews** about that brand from Twitter
- **NLTK's VADER** sentiment analyzer processes the reviews
- Results are displayed as:
  - 📈 **Pie chart** showing positive/neutral/negative sentiment distribution
  - 📝 **Top 5 positive reviews**
  - 📝 **Top 5 negative reviews**

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **AI/NLP** | OpenAI GPT-3, NLTK (VADER Sentiment) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Express |
| **Email Access** | IMAP (Gmail) |
| **Data Source** | Twitter API (via ScraperAPI) |
| **Deployment** | Heroku (Procfile configured) |

---

## 📋 Prerequisites

Before using Optim.AI, you need:

1. **Gmail App Password**
   - Enable 2-Step Verification on your Google Account
   - Generate an App Password at: `Security → 2 Step Verification → App Passwords`

2. **OpenAI API Key**
   - Sign up at: https://platform.openai.com/signup
   - Generate an API key from your dashboard

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Optim.AI

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run locally
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Deployment (Heroku)

The project is pre-configured for Heroku deployment:

```bash
# Create Heroku app
heroku create optim-ai

# Push to Heroku
git push heroku main

# Open the app
heroku open
```

---

## 📁 Project Structure

```
Optim.AI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version specification
├── Procfile            # Heroku deployment configuration
├── setup.sh            # Setup script
├── nltk.txt            # NLTK data path configuration
├── nltk/               # NLTK sentiment data
├── LICENSE             # Project license
└── README.md           # This file
```

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| **Email Extraction** | Automatically parse delivery dates, prices, and brands from emails |
| **Smart Comparison** | Identify earliest delivery and cheapest price options |
| **Brand Research** | Fetch real-time brand reviews from Twitter |
| **Sentiment Analysis** | Visual sentiment breakdown with pie charts |
| **Interactive UI** | User-friendly Streamlit interface |

---

## ⚠️ Important Notes

- The app processes up to **3 emails per session** to avoid rate limiting
- After 3 emails, there is a **60-second wait** before continuing
- Email parsing accuracy depends on the clarity of the email content
- OpenAI API usage may incur costs based on your API plan

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Akash Biswas** - Creator of Optim.AI

---

## 🔗 Links

- [OpenAI Platform](https://platform.openai.com)
- [NLTK Documentation](https://www.nltk.org)
- [Streamlit Documentation](https://streamlit.io)
- [Gmail App Passwords](https://support.google.com/accounts/answer/185833)

---

*Unlock the power of data-driven decision making with Optim.AI* 📈
