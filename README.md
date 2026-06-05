Python Code Sample by Ceily R. Addison

This project analyzes linguistic patterns in public facing government documents from the lead-up to the Iraq War (2001–2003). Using LIWC-22 output scored across 536 primary source documents, the analysis tests whether linguistic profiles differ meaningfully across institutional sources, time periods, and the later-determined accuracy of WMD claims.

Data is drawn primarily from the Fund for Independence in Journalism's Center for Public Integrity dataset, which documents public statements by eight senior Bush administration officials between September 11, 2001 and September 11, 2003. The variable name "deception" is used, however this speculative language is not representative of my personal opinion, so I recode to "accurate" and "disputed". Fewer than 30 documents are from other sources listed in IraqLIWCCombined.csv. These documents will have missing accuracy codings!

Core variables: Analytic, Clout, Authentic, Tone, moral, power

To run: clone the repository, then pip install pandas numpy scipy matplotlib seaborn statsmodels and execute LIWC_Analysis.py or open LIWC_Analysis.ipynb

Analysis Header (included in .ipynb file):

- Documents coded across three time periods: T0 (Pre-War Framing), T1 (Build-up (post Authorization for Use of Military Force Against Iraq)), T2 (Post-Invasion). Distribution: T0 n=163, T1 n=188, T2 n=185
- Statements coded as accurate or disputed based on the eventual finding that no active WMD programs were present in Iraq. Groups are imbalanced (accurate n=88, disputed n=404), coding taken directly from the original dataset.
- Prepared speech (speeches, official documents) and spontaneous speech (press briefings, interviews) analyzed separately per LIWC-22 guidance. Authenticity scoring is unreliable for prepared remarks
- All variables non-normally distributed. Non-parametric tests used throughout (Shapiro-Wilk confirmed, p < .05 across all groups)
- Kruskal-Wallis used as omnibus test for 3+ group comparisons; pairwise Mann-Whitney U with Benjamini-Hochberg correction applied where significant
- Results interpreted as stochastic dominance rather than median differences, since groups cannot be fully independent.
- Legislative documents excluded from institutional comparisons due to insufficient and structurally distinct data
- Missing accuracy for first 45 documents to be expected. They are dropped from accuracy-based analysis.
--------------------------------------------------------------------------------------------------------------------------------------------------

Key References:

Lewis, Charles, et al. "Iraq: The War Card." Fund for Independence in Journalism / Center for Public Integrity, Jan. 2008, updated 22 Mar. 2023, publicintegrity.org/politics/search-the-935-iraq-war-false-statements/. Data available at github.com/PublicI/iraq-war-card

Pennebaker, James W., et al. LIWC-22: Linguistic Inquiry and Word Count. Pennebaker Conglomerates, 2022.
Schober, Patrick, Christa Boer, and Lothar A. Schwarte. "Correlatio



All References:

Breiman, Leo. "Random Forests." Machine Learning, vol. 45, 2001, pp. 5–32. doi:10.1023/A:1010933404324.
Bush, George W. "Address to the Nation on Iraq." The American Presidency Project, edited by Gerhard Peters and John T. Woolley, www.presidency.ucsb.edu/node/212791.

Commission on the Intelligence Capabilities of the United States Regarding Weapons of Mass Destruction. Report of the Commission on the Intelligence Capabilities of the United States Regarding Weapons of Mass Destruction. 31 Mar. 2005.

Council on Foreign Relations. "The Invasion of Iraq." The 10 Best 10 Worst U.S. Foreign Policy Decisions, www.cfr.org/ten-best-ten-worst-us-foreign-policy-decisions/the-iraq-war/.

Dinno, Alexis. "Nonparametric Pairwise Multiple Comparisons in Independent Groups Using Dunn's Test." The Stata Journal, vol. 15, no. 1, 2015, pp. 292–300.

Dunn, O. J. "Multiple Comparisons Among Means." Journal of the American Statistical Association, vol. 56, 1961, pp. 52–64.

Ghasemi, A., and S. Zahediasl. "Normality Tests for Statistical Analysis: A Guide for Non-Statisticians." International Journal of Endocrinology and Metabolism, vol. 10, no. 2, 2012, pp. 486–489. doi:10.5812/ijem.3505.

Hanley, James A., and Barbara J. McNeil. "The Meaning and Use of the Area Under a Receiver Operating Characteristic (ROC) Curve." Radiology, vol. 143, no. 1, 1982, pp. 29–36. doi:10.1148/radiology.143.1.7063747.

Holland, B. S., and M. D. Copenhaver. "Improved Bonferroni-Type Multiple Testing Procedures." Scandinavian Journal of Statistics, vol. 6, 1988, pp. 65–70.

Jervis, Robert. Why Intelligence Fails: Lessons from the Iranian Revolution and the Iraq War. Cornell University Press, 2010.

Jones, Lucy S., et al. "Can Linguistic Analysis Be Used to Identify Whether Adolescents with a Chronic Illness Are Depressed?" Clinical Psychology & Psychotherapy, vol. 27, no. 2, 2020, pp. 179–192. doi:10.1002/cpp.2417.

Jordan, K. N., et al. "Examining Long-Term Trends in Politics and Culture through Language of Political Leaders and Cultural Institutions." Proceedings of the National Academy of Sciences, vol. 116, no. 9, 2019, pp. 3476–3481. doi:10.1073/pnas.1811987116.

Kruskal, W. H., and W. A. Wallis. "Use of Ranks in One-Criterion Variance Analysis." Journal of the American Statistical Association, vol. 47, 1952, pp. 583–621.

Lewis, Charles, et al. "Iraq: The War Card." Fund for Independence in Journalism / Center for Public Integrity, Jan. 2008, updated 22 Mar. 2023, publicintegrity.org/politics/search-the-935-iraq-war-false-statements/. Data available at github.com/PublicI/iraq-war-card.

LIWC-22 Descriptive Statistics: Test Kitchen Corpus. Pennebaker Conglomerates, www.liwc.app/static/documents/LIWC-22.Descriptive.Statistics-Test.Kitchen.xlsx.

Newman, Matthew L., et al. "Lying Words: Predicting Deception from Linguistic Styles." Personality and Social Psychology Bulletin, vol. 29, no. 5, 2003, pp. 665–675. doi:10.1177/0146167203029005010.

Pennebaker, James W., et al. LIWC-22: Linguistic Inquiry and Word Count. Pennebaker Conglomerates, 2022.
Schober, Patrick, Christa Boer, and Lothar A. Schwarte. "Correlation Coefficients: Appropriate Use and Interpretation." Anesthesia & Analgesia, vol. 126, no. 5, 2018, pp. 1763–1768. doi:10.1213/ANE.0000000000002864.

Strobl, Carolin, et al. "Bias in Random Forest Variable Importance Measures: Illustrations, Sources and a Solution." BMC Bioinformatics, vol. 8, 2007, p. 25. doi:10.1186/1471-2105-8-25.

Development of this code was assisted in part by AI tools for analysis replication and some syntax reference/debugging. All analytical decisions, interpretations, and results are my own. Sources for time period identification, data, and statistical method selection are included in the references.
