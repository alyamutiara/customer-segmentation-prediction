-- report_country_customer.sql
{{ config(
  materialized='table'
) }}
SELECT
    c.country,
    COUNT(fi.invoice_id) AS total_invoices,
    SUM(fi.total_price) AS total_revenue
FROM {{ ref('fct_invoices') }} fi
JOIN {{ ref('dim_customer') }} c ON fi.customer_id = c.customer_id
WHERE fi.customer_id IS NOT NULL 
  AND c.country IS NOT NULL 
  AND fi.total_price > 0
GROUP BY c.country
ORDER BY total_revenue DESC