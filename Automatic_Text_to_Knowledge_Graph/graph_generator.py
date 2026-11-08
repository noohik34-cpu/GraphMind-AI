import re
import spacy
import networkx as nx

# Load English NLP model
nlp = spacy.load("en_core_web_sm")


def normalize_entity(name):
    """Normalize entity names to reduce duplicate nodes."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def extract_relation(sentence):
    """Extract a basic relation using grammatical information."""

    verbs = [
        token for token in sentence
        if token.pos_ == "VERB"
        and token.dep_ in ("ROOT", "conj")
    ]

    if verbs:
        return verbs[0].lemma_.upper()

    return "RELATED_TO"


def calculate_confidence(sentence, entity_count, relation):
    """Heuristic score, not a calibrated probability."""

    score = 0.55

    if relation != "RELATED_TO":
        score += 0.20

    if entity_count == 2:
        score += 0.10

    if len(sentence) < 150:
        score += 0.05

    return min(round(score, 2), 0.95)


def extract_knowledge(text):

    doc = nlp(text)

    graph = nx.DiGraph()
    entities = {}

    # 1. Extract entities
    for ent in doc.ents:

        name = ent.text.strip()
        key = normalize_entity(name)

        if not key:
            continue

        if key not in entities:

            entities[key] = {
                "name": name,
                "type": ent.label_,
                "mentions": [name]
            }

            graph.add_node(
                key,
                label=name,
                entity_type=ent.label_,
                mentions=name
            )

        elif name not in entities[key]["mentions"]:

            entities[key]["mentions"].append(name)

            graph.nodes[key]["mentions"] = ", ".join(
                entities[key]["mentions"]
            )

    # 2. Extract basic sentence-level relationships
    for sentence in doc.sents:

        sentence_entities = []

        for ent in sentence.ents:

            key = normalize_entity(ent.text)

            if key in entities:
                sentence_entities.append((ent, key))

        sentence_entities.sort(
            key=lambda item: item[0].start_char
        )

        for i in range(len(sentence_entities) - 1):

            source_ent, source = sentence_entities[i]
            target_ent, target = sentence_entities[i + 1]

            if source == target:
                continue

            relation = extract_relation(sentence)

            confidence = calculate_confidence(
                sentence.text,
                len(sentence_entities),
                relation
            )

            evidence = sentence.text.strip()

            if graph.has_edge(source, target):

                edge = graph[source][target]

                if evidence not in edge["evidence"]:
                    edge["evidence"] += " | " + evidence

                edge["confidence"] = max(
                    edge["confidence"],
                    confidence
                )

            else:

                graph.add_edge(
                    source,
                    target,
                    relation=relation,
                    confidence=confidence,
                    evidence=evidence
                )

    return graph, entities


def graph_statistics(graph):

    nodes = graph.number_of_nodes()

    return {
        "entities": nodes,
        "relationships": graph.number_of_edges(),
        "density": round(nx.density(graph), 3),
        "components": (
            nx.number_weakly_connected_components(graph)
            if nodes else 0
        )
    }


def get_relationships(graph):

    records = []

    for source, target, data in graph.edges(data=True):

        records.append({
            "Source": graph.nodes[source]["label"],
            "Relationship": data.get("relation", "RELATED_TO"),
            "Target": graph.nodes[target]["label"],
            "Confidence": data.get("confidence", 0),
            "Evidence": data.get("evidence", "")
        })

    return records


def answer_question(graph, question):

    if not question.strip():
        return "Please enter a question."

    words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    )

    results = []

    for source, target, data in graph.edges(data=True):

        source_name = graph.nodes[source]["label"]
        target_name = graph.nodes[target]["label"]
        relation = data.get("relation", "RELATED_TO")

        searchable = (
            source_name.lower() + " " +
            target_name.lower() + " " +
            relation.lower().replace("_", " ")
        )

        score = sum(
            1 for word in words
            if word in searchable
        )

        if score > 0:

            results.append({
                "score": score,
                "answer": (
                    f"{source_name} → {relation} → {target_name}"
                ),
                "confidence": data.get("confidence", 0),
                "evidence": data.get("evidence", "")
            })

    results.sort(
        key=lambda item: (
            item["score"],
            item["confidence"]
        ),
        reverse=True
    )

    if not results:
        return "No matching relationship found in the graph."

    return "\n\n".join(
        (
            f"🔗 {item['answer']}\n"
            f"Confidence: {item['confidence']:.0%}\n"
            f"Evidence: {item['evidence']}"
        )
        for item in results[:10]
    )