-- fct_invoices.sql
{{ config(
  materialized='table'
) }}
-- Create the fact table by joining the relevant keys from dimension table
SELECT
  InvoiceNo AS invoice_id,
  date_part AS datetime_id,
  StockCode AS product_id,
  CustomerID AS customer_id,
  Quantity AS quantity,
  UnitPrice * Quantity AS total_price
FROM (
  SELECT
    InvoiceNo,
    StockCode,
    CustomerID,
    UnitPrice,
    Quantity,
    Country,
    {{ parse_invoice_date('InvoiceDate') }} AS datetime_invoice,
    {{ parse_invoice_date('InvoiceDate') }} AS date_part
  FROM {{ source('retail', 'raw_invoices') }}
)