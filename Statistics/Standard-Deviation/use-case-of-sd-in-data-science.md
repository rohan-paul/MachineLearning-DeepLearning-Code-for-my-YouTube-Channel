# My Youtube Link => https://youtu.be/w2WrskWX60o

### A classic use case - Remove Outliers Using Normal Distribution and S.D.

Say, we have millions of data points with significant degree of outliers. The outlier portion of the data are the ones with very high or very low values.

In this case, I need to remove these outlier values because they will make the scales on my graph unrealistic. The challenge was that the number of these outlier values will never be stable. Sometimes we would get all valid values and sometimes these erroneous data-points would cover as much as 10% of the data points.

So an approach that can be taken to remove the outlier points is by eliminating any points that aere above (Mean + 2*SD) and any points below (Mean - 2*SD) before plotting them.