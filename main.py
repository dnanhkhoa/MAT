from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from shutil import copy2
from threading import Lock
from typing import Any

import orjson
import regex
import spacy
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from spaczz.matcher import FuzzyMatcher
from torch import Tensor

app = FastAPI()

templates = Jinja2Templates(directory="templates")


class Data:
    def __init__(
        self,
        model_dir: str,
        data_dir: str,
        state_dir: str,
        backup_dir: str,
        matching_threshold: int = 100,
    ) -> None:
        self.lock = Lock()

        self.model = SentenceTransformer(model_dir, device="cpu")

        self.nlp = spacy.blank(name="en")
        self.matching_threshold = matching_threshold

        self.data_dir = Path(data_dir)
        self.state_dir = Path(state_dir)
        self.backup_dir = Path(backup_dir)

        self.data: defaultdict[
            str,
            dict[str, list[dict[str, str | list[str] | list[tuple[str, str, str]]]]],
        ] = defaultdict(dict)

        self.states: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def get_data_files(self) -> list[str]:
        return [fp.name for fp in sorted(self.data_dir.glob(pattern="*.json"))]

    def mark_entities(self, matcher: FuzzyMatcher, plain_doc: str) -> str:
        if self.matching_threshold >= 0 and plain_doc:
            doc = self.nlp(plain_doc)

            matches: dict[str, str] = {}

            def func(m: tuple[str, int, int, int, str]) -> tuple[int, int, int]:
                _, start, end, ratio, _ = m

                return (ratio, end - start, -start)

            for match_idx, (match_id, start, end, ratio, _) in enumerate(
                sorted(
                    matcher(doc),
                    key=func,
                    reverse=True,
                )
            ):
                if ratio < self.matching_threshold:
                    continue

                span = doc[start:end].text.strip()
                span_id = f"=$#@([SPAN_{match_idx}])@#$="

                plain_doc = plain_doc.replace(span, span_id)

                ent_role, rel, _ = match_id.split("_")

                matches[
                    span_id
                ] = f'<span data-role="{ent_role}" data-relation="{rel}">{span}</span>'

            for span_id, span in matches.items():
                plain_doc = plain_doc.replace(span_id, span)

        return plain_doc

    def load_data(self, data_file: str) -> None:
        with self.lock:
            data_name = data_file.lstrip(r"\/")

            if data_name in self.data:
                return None

            data_fp = self.data_dir / data_name

            if not data_fp.is_file():
                return None

            with data_fp.open(mode="rb") as f:
                for doc in map(orjson.loads, f):
                    is_extension_mode = "para_3_hop" in doc

                    doc_id = doc["_id"].strip()

                    paragraphs: list[
                        dict[str, str | list[str] | list[tuple[str, str, str]]]
                    ] = []

                    for title, sentences in (
                        [doc["para_3_hop"]] if is_extension_mode else doc["context"]
                    ):
                        paragraphs.append(
                            {
                                "title": title.strip(),
                                "sentences": list(map(str.strip, sentences)),
                            }
                        )

                    evidences: defaultdict[
                        int, dict[str, tuple[str, str, str]]
                    ] = defaultdict(dict)

                    for evidence in doc["evidences_annotation"]:
                        assert len(evidence) == 1

                        for evidence_id, (sub, rel, obj) in evidence.items():
                            evidence_id = evidence_id.strip()

                            self.states[data_name][doc_id][evidence_id] = 0

                            paragraph_idx = int(evidence_id.split("_")[-1]) - 1

                            if is_extension_mode:
                                assert len(paragraphs) == 1

                                paragraph_idx = 0

                            evidences[paragraph_idx][evidence_id] = (
                                sub.strip(),
                                rel.strip(),
                                obj.strip(),
                            )

                    for paragraph_idx, paragraph in enumerate(paragraphs):
                        ents: dict[str, str] = {}

                        rels: list[str] = []

                        for sub, rel, obj in evidences[paragraph_idx].values():
                            ents[sub] = f"S_{rel}"
                            ents[obj] = f"E_{rel}"

                            rels.append(rel)

                        matcher = FuzzyMatcher(vocab=self.nlp.vocab)

                        for ent_idx, ent in enumerate(
                            sorted(ents, key=len, reverse=True)
                        ):
                            matcher.add(  # type: ignore
                                label=f"{ents[ent]}_{ent_idx}",
                                patterns=[self.nlp(ent)],
                            )

                        sentences: list[str] = paragraph["sentences"]  # type: ignore

                        paragraph["paragraph"] = " ".join(
                            [
                                f'<span data-highlightable="1" '
                                f'data-sentence-id="{paragraph_idx}_{sentence_idx}">'
                                f"{self.mark_entities(matcher, sentence)}"
                                "</span>"
                                for sentence_idx, sentence in enumerate(sentences)
                            ]
                        )

                        rel_embeddings: Tensor = self.model.encode(  # type: ignore
                            rels, convert_to_tensor=True, normalize_embeddings=True
                        )

                        sentence_embeddings: Tensor = self.model.encode(  # type: ignore
                            sentences, convert_to_tensor=True, normalize_embeddings=True
                        )

                        rel_sentence_indices: list[int] = (
                            util.dot_score(rel_embeddings, sentence_embeddings)
                            .argmax(dim=1)
                            .tolist()  # type: ignore
                        )

                        paragraph["evidences"] = [
                            (
                                evidence_id,
                                ", ".join(
                                    (
                                        self.mark_entities(matcher, plain_doc=sub),
                                        f"<code>{rel}</code>",
                                        self.mark_entities(matcher, plain_doc=obj),
                                    )
                                ),
                                f"{paragraph_idx}_{rel_sentence_indices[evidence_idx]}",
                            )
                            for evidence_idx, (
                                evidence_id,
                                (sub, rel, obj),
                            ) in enumerate(evidences[paragraph_idx].items())
                        ]

                        def func(e: tuple[str, str, str]) -> int:
                            evidence_id, *_ = e

                            return int(evidence_id.split("_")[-1])

                        paragraph["evidences"] = sorted(
                            paragraph["evidences"], key=func
                        )

                    if not is_extension_mode:
                        evidence_id = f"{doc_id}_ans"
                        evidence = doc["answer"].strip()

                        paragraphs.append(
                            {
                                "title": "Question and Answer",
                                "paragraph": doc["question"].strip(),
                                "evidences": [(evidence_id, evidence, "_")],
                            }
                        )

                        self.states[data_name][doc_id][evidence_id] = 0

                    self.data[data_name][doc_id] = paragraphs

            state_fp = self.state_dir / data_name

            if not state_fp.is_file():
                return None

            self.backup_dir.mkdir(parents=True, exist_ok=True)

            backup_fp = self.backup_dir / data_name

            backup_suffix = "_".join(regex.split(r"[\s[:punct:]]", str(datetime.now())))

            backup_fp = backup_fp.with_stem(f"{backup_fp.stem}_{backup_suffix}")

            copy2(src=state_fp, dst=backup_fp)

            with state_fp.open(mode="rb") as f:
                for doc_id, evidence_ids in orjson.loads(f.read()).items():
                    for evidence_id, evidence_ans in evidence_ids.items():
                        self.states[data_name][doc_id][evidence_id] = evidence_ans

    def get_doc_ids(self, data_file: str) -> list[tuple[float, str]] | None:
        self.load_data(data_file)

        with self.lock:
            data_name = data_file.lstrip(r"\/")

            if data_name not in self.states:
                return None

            doc_ids: list[tuple[float, str]] = []

            for doc_id, evidence_ids in self.states[data_name].items():
                counts = Counter(evidence_ids.values())

                total_answers = sum(counts.values())

                progress = round(
                    100 - (total_answers and counts[0] * 100.0 / total_answers),
                    ndigits=2,
                )

                doc_ids.append((progress, doc_id))

            return doc_ids

    def get_doc(
        self, data_file: str, doc_id: str
    ) -> (
        dict[
            str,
            str
            | float
            | list[dict[str, str | list[str] | list[tuple[str, str, str]]]]
            | dict[str, int],
        ]
        | None
    ):
        with self.lock:
            data_name = data_file.lstrip(r"\/")

            if data_name not in self.states:
                return None

            evidence_ids = self.states[data_name][doc_id]

            counts = Counter(evidence_ids.values())

            total_answers = sum(counts.values())

            progress = round(
                100 - (total_answers and counts[0] * 100.0 / total_answers),
                ndigits=2,
            )

            doc = {
                "id": doc_id,
                "progress": progress,
                "paragraphs": self.data[data_name][doc_id],
                "answers": self.states[data_name][doc_id],
            }

            return doc

    def save_states(self, data_file: str) -> None:
        with self.lock:
            data_name = data_file.lstrip(r"\/")

            if data_name not in self.states:
                return None

            self.state_dir.mkdir(parents=True, exist_ok=True)

            state_fp = self.state_dir / data_name

            with state_fp.open(mode="wb") as f:
                f.write(orjson.dumps(self.states[data_name]))

    def save_answer(
        self, data_file: str, doc_id: str, evidence_id: str, evidence_ans: int
    ) -> (
        dict[
            str,
            str
            | float
            | list[dict[str, str | list[str] | list[tuple[str, str, str]]]]
            | dict[str, int],
        ]
        | None
    ):
        with self.lock:
            data_name = data_file.lstrip(r"\/")

            if data_name not in self.states:
                return None

            self.states[data_name][doc_id][evidence_id] = evidence_ans

        self.save_states(data_file)

        return self.get_doc(data_file, doc_id)


data = Data(
    model_dir="./models/all-MiniLM-L6-v2",
    data_dir="data",
    state_dir="answers",
    backup_dir="backups",
    matching_threshold=80,
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse(  # type: ignore
        "index.html", context={"request": request, "data_files": data.get_data_files()}
    )


@app.get("/ids/", response_class=ORJSONResponse)
def get_doc_ids(data_file: str) -> Any:
    return {"data": data.get_doc_ids(data_file)}


@app.get("/doc/", response_class=HTMLResponse)
def get_doc(request: Request, data_file: str, doc_id: str) -> Any:
    return templates.TemplateResponse(  # type: ignore
        "doc.html", context={"request": request, "doc": data.get_doc(data_file, doc_id)}
    )


@app.put("/answer/", response_class=ORJSONResponse)
def put_answer(data_file: str, doc_id: str, evidence_id: str, evidence_ans: int) -> Any:
    return data.save_answer(data_file, doc_id, evidence_id, evidence_ans)
