{{ config(
  materialized='table'
) }}
SELECT
  DISTINCT fct.customer_id,
  fct.datetime_id,
  cus.Country,
  COUNT(fct.invoice_id) AS Total_Orders,
  SUM(fct.quantity) AS Total_Items,
  SUM(fct.total_price) AS Total_Sales,
  rfm.RFM_Score,
  rfm.RFM_Segment,
  rfm.Segment
FROM {{ ref('fct_invoices') }} AS fct
LEFT JOIN {{ source('retail', 'rfm_segments_final') }} AS rfm
  ON fct.customer_id = rfm.CustomerID
LEFT JOIN {{ ref('dim_customer') }} AS cus
  ON fct.customer_id = cus.customer_id
WHERE fct.customer_id IS NOT NULL
  AND fct.datetime_id IS NOT NULL
  AND fct.total_price > 0
  AND fct.quantity > 0
GROUP BY
  fct.datetime_id,
  fct.customer_id,
  cus.Country,
  rfm.RFM_Score,
  rfm.RFM_Segment,
  rfm.Segment