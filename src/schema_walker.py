class SchemaWalker:
    def __init__(self, schema):
        self.schema = schema

    def walk(self):
        results = []
        self._walk_dict(self.schema, [], results)
        return results

    def _walk_dict(self, node, path, results):
        for key, value in node.items():
            new_path = path + [key]
            if isinstance(value, dict):
                self._walk_dict(value, new_path, results)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._walk_dict(item, new_path, results)
            if not isinstance(value, (dict, list)):
                path_str = '.'.join(new_path)
                results.append((path_str, value))