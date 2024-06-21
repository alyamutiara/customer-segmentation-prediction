
SELECT
  fct.datetime_id,
  fct.customer_id,
  cus.Country,
  COUNT(fct.invoice_id) AS Total_Orders,
  SUM(fct.quantity) AS Total_Items,
  SUM(fct.total_price) AS Total_Sales,
  rfm.RFM_Segment,
  rfm.Segment
FROM `iconic-indexer-418610`.`retail`.`fct_invoices` AS fct
LEFT JOIN `iconic-indexer-418610`.`retail`.`rfm_segments_final` AS rfm
  ON CAST(fct.customer_id AS INT64) = rfm.CustomerID
LEFT JOIN `iconic-indexer-418610`.`retail`.`dim_customer` AS cus
  ON CAST(fct.customer_id AS FLOAT64) = cus.customer_id
GROUP BY
  fct.datetime_id,
  fct.customer_id,
  cus.Country,
  rfm.RFM_Segment,
  rfm.Segment