-- dim_datetime.sql
{{ config(
  materialized='table'
) }}
-- Create a CTE to extract date and time components
WITH datetime_cte AS (  
  SELECT
    {{ parse_invoice_date('InvoiceDate') }} AS datetime_invoice,
    {{ parse_invoice_date('InvoiceDate') }} AS date_part,
  FROM {{ source('retail', 'raw_invoices') }}
)
SELECT
  date_part as datetime_id,
  EXTRACT(YEAR FROM date_part) AS year,
  EXTRACT(MONTH FROM date_part) AS month,
  EXTRACT(DAY FROM date_part) AS day,
  EXTRACT(HOUR FROM date_part) AS hour,
  EXTRACT(QUARTER FROM date_part) AS quarter,
  EXTRACT(ISOWEEK FROM date_part) AS week_of_year,
  EXTRACT(DAYOFWEEK FROM date_part) AS day_of_week
FROM datetime_cte