import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_simd_closure import changed

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
GIT = "git+https://example.com/foo"


def package(name, version, source=None, checksum=None, deps=()):
    lines = ["[[package]]", f'name = "{name}"', f'version = "{version}"']
    if source:
        lines.append(f'source = "{source}"')
    if checksum:
        lines.append(f'checksum = "{checksum}"')
    if deps:
        lines.append("dependencies = [" + ", ".join(f'"{d}"' for d in deps) + "]")
    return "\n".join(lines)


def lock(*packages):
    return "version = 4\n\n" + "\n\n".join(packages) + "\n"


def base_packages(blst_checksum="aa", rev="1111", unrelated="1.0.0"):
    return [
        package(
            "silver_crypto_simd",
            "0.1.0",
            deps=["blst", f"foo 1.0.0 ({GIT})", f"foo 1.0.0 ({GIT}2)"],
        ),
        package("blst", "0.3.16", REGISTRY, blst_checksum, deps=["cc"]),
        package("cc", "1.2.0", REGISTRY, "cc"),
        package("foo", "1.0.0", f"{GIT}#{rev}"),
        package("foo", "1.0.0", f"{GIT}2#9999"),
        package("tokio", unrelated, REGISTRY, "dd"),
    ]


class Closure(unittest.TestCase):
    def check(self, base, head):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, text in (("base", base), ("head", head)):
                path = os.path.join(d, f"{name}.lock")
                with open(path, "w") as f:
                    f.write(text)
                paths.append(path)
            return changed(*paths)

    def test_identical_lockfiles_are_unchanged(self):
        self.assertFalse(self.check(lock(*base_packages()), lock(*base_packages())))

    def test_package_outside_closure_is_ignored(self):
        head = lock(*base_packages(unrelated="1.1.0"))
        self.assertFalse(self.check(lock(*base_packages()), head))

    def test_registry_dependency_change_counts(self):
        head = lock(*base_packages(blst_checksum="bb"))
        self.assertTrue(self.check(lock(*base_packages()), head))

    def test_git_revision_of_source_qualified_dependency_counts(self):
        head = lock(*base_packages(rev="2222"))
        self.assertTrue(self.check(lock(*base_packages()), head))

    def test_transitive_dependency_change_counts(self):
        packages = base_packages()
        packages[2] = package("cc", "1.2.1", REGISTRY, "cc")
        packages[1] = package("blst", "0.3.16", REGISTRY, "aa", deps=["cc"])
        self.assertTrue(self.check(lock(*base_packages()), lock(*packages)))

    def test_unresolved_edge_counts(self):
        packages = base_packages()
        packages[0] = package("silver_crypto_simd", "0.1.0", deps=["missing"])
        self.assertTrue(self.check(lock(*packages), lock(*packages)))

    def test_ambiguous_edge_counts(self):
        packages = base_packages()
        packages[0] = package("silver_crypto_simd", "0.1.0", deps=["foo"])
        self.assertTrue(self.check(lock(*packages), lock(*packages)))

    def test_root_missing_at_base_counts(self):
        base = lock(*base_packages()[1:])
        self.assertTrue(self.check(base, lock(*base_packages())))


if __name__ == "__main__":
    unittest.main()
