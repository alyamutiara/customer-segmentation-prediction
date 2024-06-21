-- dim_customer.sql
{{ config(
  materialized='table'
) }}
-- Create the dimension table
WITH datetime_cte AS (  
  SELECT
    CustomerID,
    Country,
    InvoiceNo,
    UnitPrice,
    Quantity,
    InvoiceDate,
    {{ parse_invoice_date('InvoiceDate') }} AS datetime_invoice,
    {{ parse_invoice_date('InvoiceDate') }} AS date_part,
  FROM {{ source('retail', 'raw_invoices') }}
)

SELECT
  CustomerID as customer_id,
  Country,
  MIN(date_part) AS first_transaction,
  MAX(date_part) AS last_transaction,
  COUNT(InvoiceNo) AS count_order
FROM datetime_cte
GROUP BY CustomerID, Country