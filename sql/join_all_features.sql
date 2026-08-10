-- join_all_features.sql
-- Final feature matrix: LEFT JOIN all aggregated feature tables onto application_train
-- Assumes these tables already exist: bureau_features, previous_app_features,
-- installments_features, pos_cash_features, credit_card_features
-- ============================================================

SELECT
    app.*,

    -- Bureau features
    bf.BUREAU_CREDIT_COUNT,
    bf.BUREAU_ACTIVE_COUNT,
    bf.BUREAU_CLOSED_COUNT,
    bf.BUREAU_ACTIVE_RATIO,
    bf.BUREAU_AVG_CREDIT_SUM,
    bf.BUREAU_MAX_CREDIT_SUM,
    bf.BUREAU_AVG_CREDIT_DEBT,
    bf.BUREAU_MAX_OVERDUE,
    bf.BUREAU_AVG_DAYS_CREDIT,
    bf.BUREAU_CREDIT_DAY_OVERDUE_MAX,
    bf.BUREAU_AVG_CREDIT_DURATION,
    bf.BUREAU_PROLONG_COUNT,
    bf.BUREAU_BAL_DPD_MONTHS,
    bf.BUREAU_BAL_MAX_DPD,
    bf.BUREAU_BAL_STATUS_C_RATIO,

    -- Previous application features
    pf.PREV_APP_COUNT,
    pf.PREV_APPROVED_COUNT,
    pf.PREV_REFUSED_COUNT,
    pf.PREV_APPROVAL_RATE,
    pf.PREV_AVG_AMT_APPLICATION,
    pf.PREV_AVG_AMT_CREDIT,
    pf.PREV_AVG_CREDIT_TO_APP_RATIO,
    pf.PREV_AVG_DOWN_PAYMENT,
    pf.PREV_AVG_DAYS_DECISION,
    pf.PREV_CASH_COUNT,
    pf.PREV_CONSUMER_COUNT,
    pf.PREV_AVG_CNT_PAYMENT,
    pf.PREV_REPEATER_RATIO,

    -- Installment features
    inf.INST_COUNT,
    inf.INST_LATE_PAYMENT_COUNT,
    inf.INST_LATE_PAYMENT_RATIO,
    inf.INST_AVG_DAYS_LATE,
    inf.INST_MAX_DAYS_LATE,
    inf.INST_AVG_PAYMENT_DIFF,
    inf.INST_MAX_PAYMENT_SHORTFALL,
    inf.INST_PAYMENT_SHORTFALL_RATIO,
    inf.INST_AVG_PAYMENT,

    -- POS/Cash balance features
    pcf.POS_COUNT,
    pcf.POS_ACTIVE_COUNT,
    pcf.POS_COMPLETED_COUNT,
    pcf.POS_MAX_DPD,
    pcf.POS_AVG_DPD,
    pcf.POS_DPD_MONTHS,
    pcf.POS_DPD_RATIO,
    pcf.POS_MAX_DPD_DEF,
    pcf.POS_AVG_INSTALMENT_FUTURE,

    -- Credit card features
    ccf.CC_COUNT,
    ccf.CC_AVG_BALANCE,
    ccf.CC_MAX_BALANCE,
    ccf.CC_AVG_CREDIT_LIMIT,
    ccf.CC_AVG_UTILIZATION,
    ccf.CC_MAX_UTILIZATION,
    ccf.CC_AVG_DRAWINGS,
    ccf.CC_AVG_ATM_DRAWINGS,
    ccf.CC_AVG_PAYMENT,
    ccf.CC_MAX_DPD,
    ccf.CC_DPD_MONTHS,
    ccf.CC_AVG_MIN_INSTALMENT

FROM application_train app
LEFT JOIN bureau_features bf      ON app.SK_ID_CURR = bf.SK_ID_CURR
LEFT JOIN previous_app_features pf ON app.SK_ID_CURR = pf.SK_ID_CURR
LEFT JOIN installments_features inf ON app.SK_ID_CURR = inf.SK_ID_CURR
LEFT JOIN pos_cash_features pcf    ON app.SK_ID_CURR = pcf.SK_ID_CURR
LEFT JOIN credit_card_features ccf ON app.SK_ID_CURR = ccf.SK_ID_CURR
