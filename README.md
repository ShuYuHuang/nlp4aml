# nlp4aml
We want to build a system to detect potential targets for anti-money laundering (AML) that appear in any given Chinese article. This toolbox includes :

Background categorization: Detect if an article is related to money laundering. -(Training/Application)Word segmentation, filtering for meaningful words. -(Training)Forming word vector space for categorizing if the article is related to money laundering. -(Application)Determine if an article is related to money laundering using distance in word vector space.
Entity categorization: Detect the potential targets for AML: -(Training/Application)Identify named entities in the articles related to ML -(Training)Train an ML model to identify targets for AML among all named entities -(Application)Determine if the named entity is a target for AML
