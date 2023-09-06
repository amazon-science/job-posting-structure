# job-posting-structure

Parses structured information from HTML-formatted job postings.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

## JobStruct class

The primary class is called JobStruct and can be initialized from
a filename, an HTML string, or an existing BeautifulSoup object that
contains parsed HTML:

    j = JobStruct.from_file("myJobPosting.html")

    with open("myJobPosting.html") as f:
        posting_html_str = f.read()
    j = JobStruct.from_string(posting_html_str)

    posting_soup_obj = BeautifulSoup(posting_html_str, "html.parser")
    j = JobStruct.from_soup(posting_soup_obj)

Once initialized, the JobStruct object has attributes for each segment
that was parsed from the job posting:

* description
* benefits
* qualitifications
* responsibilities
* requirements
* eeo (Equal Employment Opportunity)
* other
