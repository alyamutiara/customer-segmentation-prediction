# Automated Customer Segmentation Prediction Pipeline
IYKRA Data Fellowship 12 capstone project for Group 2: Automated Customer Lifetime Value Prediction with Machine Learning.

<b>Driven Output</b>
- Exploration of the importance of CLV in strategic decision-making within banks.
- Highlighting the construction of a scalable data pipeline for continuous CLV model training and prediction.
- Description of the methodology used to collect, preprocess, and analyze customer data for CLV prediction.
- Insights into the machine learning algorithms tested, selection rationale, and performance metrics.
- Case studies or examples of how CLV predictions have informed marketing strategies and product offerings.
- Guidelines for banks to implement and leverage CLV prediction models.
- Design suggestions for the utilization of GenAI for related use case

## Background
There’s a Pareto principle that comes from an Italian economist saying that for many outcomes, roughly 80% of consequences comes from 20% of causes. Similarly, 80% of companies business comes from 20% customers. That’s why companies need to identify those top-valued customers and maintain the relationship with them to ensure continues revenue. In order to maintain a long-term relationship with customers, company need to schedule loyalty schemes such as the discount, offers, coupons, bonus point, gifts, etc.

Customer Lifetime Value (CLTV) helps business identify the high-potential customers. By analyzing past customer behaviour and revenue, CLTV can predicts the total value a customer can bring over time. This allow companies to prioritize efforts toward attracting and retaining customers with the greatest future profit potential.

### Problem Statement
An e-commerce company has a large number of customers to serve and wants to effectively manage personalized marketing to improve customer loyalty and decrease churning rate.

### Goal
Create a data pipeline from existing data that can be utilized to segment the current customers’ value and predict the segment for the next period.

### Objective
1. <b>Predict Customer Lifetime Value</b>: Develop a machine learning model that can predict the future value a customer will bring to the company.
2. <b>Automate the Prediction Process</b>: Create an end-to-end automated pipeline that handles data collection, preprocessing, model training, and prediction.
3. <b>Dashboard for Reporting</b>: Build a dashboard to improve data visualization and analysis for better marketing decisions.

## Proposed Architecture
![Pipeline Architecture](/img/pipeline-architecture.png)
## Data Pipeline
![image](https://github.com/alyamutiara/customer-segmentation-prediction/assets/128980804/3801b6ef-10dd-4617-b17d-67e4015ed3a6)
## Machine Learning Pipeline
![image](https://github.com/alyamutiara/customer-segmentation-prediction/assets/128980804/c04fa810-38f2-4d47-8b70-baf57618adf7)

# Data Warehouse

## Data Modeling
![Data Modeling](/img/database-scema-medalion-architectur.png)
## Data Lineage
![Data Lineage](/img/Data-lineage.png)

## Tools
- Data Warehouse: BigQuery
- Compute Engine: GCP VM Instance
- Container: Docker
- Workflow Orchestrator: Astronomer for Airflow
- Data Transformation: data build tool (dbt)
- Data Governance: Soda
- Exploratory Data Analysis: Vertex Colab Notebook
- Data Visualization: Looker Data Studio
- API: FastAPI

## Data Visualization
![Dashboard Page 1](/img/dashboard-1.png)
![Dashboard Page 2](/img/dashboard-2.png)
![Dashboard Page 0](/img/dashboard-0.png)

## Resource
- Dataset: [UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- Deck: [here](https://www.canva.com/design/DAGHz94GfKk/-6ouiyRnokpBEuaeVdSHoQ/view?utm_content=DAGHz94GfKk&utm_campaign=designshare&utm_medium=link&utm_source=editor)
- Whitepaper: [here](https://www.canva.com/design/DAGIB9v79U4/DLTx8i45HcL0BPOzoMRvbw/edit)
- API: [FastAPI](https://github.com/fahernkhan/Customer-Lifetime-Value-API)
- Dashboard: [Looker Data Studio](https://lookerstudio.google.com/u/0/reporting/736a49b1-f5cf-474a-b801-498781dbc41d/page/N5J2D)

## Team Member
- Alivia Rahma Sakina ([Github](https://github.com/Aliviarahma), [Linkedin](https://id.linkedin.com/in/aliviarahma))
- Alya Mutiara Firdausyi ([Github](https://github.com/alyamutiara), [Linkedin](https://www.linkedin.com/in/alyamf/))
- Fathurrahman Hernanda Khasan ([Github](https://github.com/fahernkhan), [Linkedin](https://id.linkedin.com/in/fathurrahmanhernanda/in))
- Muhammad Zakie Arfiansyah ([Github](https://github.com/zakiearfi), [Linkedin](https://github.com/zakiearfi))
- Ridho Alfayet Umar ([Github](https://github.com/fayetumar), [Linkedin](https://id.linkedin.com/in/ridhoal))
