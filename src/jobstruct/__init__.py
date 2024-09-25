# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
`jobstruct` provides GenAI methods for structuring information
and modeling skills and occupations in job postings.
"""

from .jobstructai      import JobStructAI
from .jobstructhtml    import JobStructHTML
from .prompts          import Prompts
from .skillsnode       import SkillsNode
from .skillstaxonomyai import SkillsTaxonomyAI

__version__ = "0.1.1"
