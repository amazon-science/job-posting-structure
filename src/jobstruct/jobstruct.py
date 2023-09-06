# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from bs4 import BeautifulSoup

class JobStruct:
    """
    A class that represents a parsed HTML job posting, starting
    from either a filename, HTML text, or a BeautifulSoup object.
    The parsed segments of the job posting, available as attributes, are:
    * description
    * benefits
    * qualitifications
    * responsibilities
    * requirements
    * eeo (Equal Employment Opportunity)
    * other
    """

    __TAGS = [
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    ]

    __SEGMENTS = {
        "description": frozenset((
            "description",
            "overview",
            "glance",
            "summary",
            "posting"
        )),
        "benefits": frozenset((
            "perks",
            "benefits",
            "offer"
        )),
        "qualifications": frozenset((
            "experience",
            "qualification",
            "qualifications",
            "skills",
        )),
        "responsibilities": frozenset((
            "responsibilities",
            "duties",
            "functions",
            "function(s)"
        )),
        "requirements": frozenset((
            "requirements",
            "required",
            "requirement"
        )),
        "eeo": frozenset((
            "equal",
            "opportunity",
            "employer"
        ))
    }

    def __init__(self, soup: BeautifulSoup = None):
        """
        Segments the HTML job posting `soup` that has been parsed by BeautifulSoup,
        and provides the segments as attributes. If no `soup` is provided, returns
        an empty structure.
        """
        self._init_segments()
        if soup is not None:
            self.soup: BeautifulSoup = soup
            self._segment()
        self._add_attributes()


    @classmethod
    def from_file(cls, filename: str) -> "JobStruct":
        """
        Creates a JobStruct object from the HTML in `filename`.
        """
        with open(filename) as f:
            soup: BeautifulSoup = BeautifulSoup(f.read(), "html.parser")
        return cls(soup)


    @classmethod
    def from_string(cls, html: str) -> "JobStruct":
        """
        Creates a JobStruct object from a `string` containing HTML.
        """
        soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        return cls(soup)


    @classmethod
    def from_soup(cls, soup: BeautifulSoup) -> "JobStruct":
        """
        Creates a JobStruct object from BeautifulSoup-parsed HTML in `soup`.
        """
        return cls(soup)


    def to_dict(self):
        """
        Convert the JobStruct object to a dictionary containing the segment
        attributes.
        """
        return {segment: list(values) for segment, values in self.segments.items()}


    def _init_segments(self):
        """
        Initial empty list for each segment type.
        """
        self.segments = {segment: list() for segment in JobStruct.__SEGMENTS.keys()}
        # Other is the catch-all type for segments that don't match a keyword.
        self.segments["other"] = list()


    def _segment(self):
        """
        Loop over HTML elements to find headings for each segment type and
        append the elements following the heading to the segment lists.
        """
        segment = "other"
        for element in self.soup.body.find_all(JobStruct.__TAGS):
            text = element.get_text(separator="\n").strip()
            if text:
                if len(text.split()) <= 5:
                    segment = self._classify_segment(text.lower())
                elif self._is_terminal(element):
                    for line in text.split("\n"):
                        if "equal opportunity employer" in line:
                            self.segments["eeo"].append(line)
                        else:
                            self.segments[segment].append(line)


    def _classify_segment(self, text: str):
        """
        Classify `text` into one of the segment types using keywords.
        Defaults to "other" if no keywords were found.
        """
        for segment, keywords in JobStruct.__SEGMENTS.items():
            if any(word.strip(":") in keywords for word in text.split()):
                return segment
        return "other"


    def _is_terminal(self, element):
        """
        """
        return all(
            element.find(tag) is None
            for tag in JobStruct.__TAGS
        )


    def _add_attributes(self):
        """
        Add attributes for each segment type to the returned object.
        """
        for segment in self.segments.keys():
            assert not hasattr(self, segment)
            setattr(self, segment, self.segments[segment])


    def __str__(self):
        output = []
        for segment, values in self.segments.items():
            if segment != "other":
                if values:
                    output.append(f"{segment}: [")
                    for value in values:
                        output.append(value)
                    output.append("]")
                else:
                    output.append(f"{segment}: []")
        return "\n".join(output)
