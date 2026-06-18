import pytest
import tempfile
import os
from pathlib import Path
from ticket_classifier.text_reader import TextReader


class TestTextReader:
    @pytest.fixture
    def reader(self):
        return TextReader()

    @pytest.fixture
    def temp_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is a test ticket content.\nIt has multiple lines.")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("text,category\n")
            f.write("First ticket text,A\n")
            f.write("Second ticket text,B\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                file_path = Path(temp_dir) / f"ticket_{i}.txt"
                file_path.write_text(f"Ticket content {i}", encoding="utf-8")
            yield temp_dir

    def test_read_txt_file(self, reader, temp_txt_file):
        content = reader.read_file(temp_txt_file)
        assert "test ticket content" in content
        assert "multiple lines" in content

    def test_read_csv_file(self, reader, temp_csv_file):
        content = reader.read_file(temp_csv_file)
        assert "First ticket text" in content
        assert "Second ticket text" in content

    def test_read_file_not_found(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.read_file("/nonexistent/file.txt")

    def test_read_file_not_a_file(self, reader, temp_dir):
        with pytest.raises(ValueError, match="Not a file"):
            reader.read_file(temp_dir)

    def test_read_unsupported_format(self, reader):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"test")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                reader.read_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_read_directory(self, reader, temp_dir):
        tickets = reader.read_directory(temp_dir)
        assert len(tickets) == 3
        assert tickets[0]["file_name"] == "ticket_0.txt"
        assert tickets[0]["content"] == "Ticket content 0"
        assert "id" in tickets[0]

    def test_read_directory_not_found(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.read_directory("/nonexistent/dir")

    def test_read_directory_not_a_dir(self, reader, temp_txt_file):
        with pytest.raises(ValueError, match="Not a directory"):
            reader.read_directory(temp_txt_file)

    def test_read_directory_with_pattern(self, reader, temp_dir):
        tickets = reader.read_directory(temp_dir, pattern="ticket_1*.txt")
        assert len(tickets) == 1
        assert tickets[0]["file_name"] == "ticket_1.txt"

    def test_empty_directory(self, reader):
        with tempfile.TemporaryDirectory() as temp_dir:
            tickets = reader.read_directory(temp_dir)
            assert len(tickets) == 0
