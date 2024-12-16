from plotnine import *
import pandas as pd

# Load the dataset
# Replace 'songs_cleaned.csv' with the path to your actual CSV file
data = pd.read_csv("songs_cleaned.csv")

attribute = 'family/spiritual'

# Replace problematic characters in the attribute string
safe_attribute = attribute.replace("/", "_").replace(" ", "_")

# Create the plot
plot = (
    ggplot(data, aes(x=f"{attribute}")) + 
    geom_histogram(fill="skyblue", color="black", binwidth=0.025) +
    labs(
        title=f"Histogram of {attribute} Label Values",
        x="Labels",
        y="Frequency"
    )
)

# Save the plot with the sanitized filename
plot.save(filename=f"{safe_attribute}_range.png")

print(f"Plot saved as {safe_attribute}_range.png")
