# Tableau calculated fields: Bank Customer Churn

Create each one with **Analysis → Create Calculated Field**, name it exactly as shown, and paste the formula.
Tableau shows CSV columns in title case (`new_cases` becomes *New Cases*). If yours show the raw names, use those in the formulas.

## Fields

### Churn Rate

```
SUM([Exited]) / COUNT([Customer Id])
```

### Overall Churn Rate

```
{ FIXED : AVG([Exited]) }
```

Reference line on every churn-rate bar.

### Balance Lost to Churn

```
SUM(IF [Exited] = 1 THEN [Balance] END)
```

### At-Risk Customer

```
[Exited] = 0 AND ([Risk Band] = "High" OR [Risk Band] = "Very high")
```

Boolean filter for the retention target list.

### Active Member Label

```
IF [Is Active Member] = 1 THEN "Active" ELSE "Inactive" END
```
