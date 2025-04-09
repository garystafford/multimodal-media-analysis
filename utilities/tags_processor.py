class TagsProcessor:
    def process_tags(self, tags: str) -> list:
        """
        Process a string of comma-delimited tags, remove duplicates, and sort alphabetically.

        Args:
            tags (str): A string of comma-delimited tags.

        Returns:
            list: A sorted list of unique tags.
        """
        try:
            # Split the string into a list of tags
            phrase_list = tags.split(",")
        except Exception as e:
            print(f"Error splitting tags: {e}")
            return []

        # Remove leading/trailing whitespace, double quotes, and periods from each phrase
        phrase_list = [
            phrase.lower().strip().replace('"', "").replace(".", "")
            for phrase in phrase_list
        ]

        # Remove items that are less than three characters long
        phrase_list = [phrase for phrase in phrase_list if len(phrase) >= 3]

        # Remove duplicates by converting to a set
        unique_tags = set(phrase_list)

        # Sort the set alphabetically and convert back to a list
        sorted_tags = sorted(unique_tags)

        return sorted_tags
