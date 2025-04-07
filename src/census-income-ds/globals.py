# Ordered list of column names of both training and test sets
COLUMN_NAMES = [
    "age",
    "class_of_worker",
    "industry_code",
    "occupation_code",
    "education",
    "wage_per_hour",
    "enrolled_in_edu_inst_last_wk",
    "marital_status",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "divdends_from_stocks",
    "tax_filer_status",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "instance_weight",
    "migration_code_change_in_msa",
    "migration_code_change_in_reg",
    "migration_code_move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "target",
]

# Numerical columns to be dropped for either information redundancy
# or lack of information (check notebook)
COLS_DROP_NUMERICAL = [
    "industry_code",
    "occupation_code",
    "own_business_or_self_employed",
    "veterans_benefits",
    "wage_per_hour",
    "year",
]

# Nominal columns to be dropped for either one dominant category,
# lack of information, or fraction of missing values (check notebook)
COLS_DROP_NOMINAL = [
    "fill_inc_questionnaire_for_veterans_admin",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "family_members_under_18",
    "migration_prev_res_in_sunbelt",
    "enrolled_in_edu_inst_last_wk",
]

# Ordered fields of the education feature, needed for ordinal
# encoding
ORDERED_EDU_LEVELS = [
    'Children',
    'Less_than_1st_grade',
    '1st_2nd_3rd_or_4th_grade',
    '5th_or_6th_grade',
    '7th_and_8th_grade',
    '9th_grade',
    '10th_grade',
    '11th_grade',
    '12th_grade_no_diploma',
    'High_school_graduate',
    'Some_college_but_no_degree',
    'Associates_degree-occup_/vocational',
    'Associates_degree-academic_program',
    'Bachelors_degree(BA_AB_BS)',
    'Masters_degree(MA_MS_MEng_MEd_MSW_MBA)',
    'Prof_school_degree_(MD_DDS_DVM_LLB_JD)',
    'Doctorate_degree(PhD_EdD)',
]

# Clusters of correlated features after encoding (check notebook)
CORRELATED_COLS_AFTER_ENCODING = [
    ["major_industry_code_Not_in_universe_or_children", "class_of_worker_Not_in_universe"],
    ["citizenship_Native-_Born_in_the_United_States", "born_in_us"],
    ["live_in_this_house_1_year_ago_No",
    "state_of_previous_residence_Not_in_universe",
    "state_of_previous_residence_Other",
    "migration_code_change_in_msa",
    "migration_code_change_in_reg",
    "migration_code_move_within_reg",],
]
