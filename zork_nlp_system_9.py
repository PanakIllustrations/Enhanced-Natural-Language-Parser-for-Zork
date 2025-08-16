#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Natural Language Parser for Zork
Incorporating techniques from:
1. Fulda et al. (2017) - Affordance Extraction via Word Embeddings
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class ActionType(Enum):
    MOVEMENT = "movement"
    TAKE = "take"
    DROP = "drop"
    EXAMINE = "examine"
    USE = "use"
    OPEN = "open"
    CLOSE = "close"
    ATTACK = "attack"
    INVENTORY = "inventory"
    LOOK = "look"
    READ = "read"
    UNLOCK = "unlock"
    PUT = "put"
    GIVE = "give"
    TALK = "talk"
    EAT = "eat"
    DRINK = "drink"
    LIGHT = "light"
    EXTINGUISH = "extinguish"

    THROW = "throw"
    SAVE = "save"
    RESTORE = "restore"
    RESTART = "restart"
    VERBOSE = "verbose"
    BRIEF = "brief"
    SUPERBRIEF = "superbrief"
    SCORE = "score"
    DIAGNOSE = "diagnose"
    QUIT = "quit"
    WAIT = "wait"
    AGAIN = "again"

    UNKNOWN = "unknown"


@dataclass
class GameObject:
    """Represents an object with affordances (Fulda et al. 2017)"""
    name: str
    affordances: Set[str] = field(default_factory=set)
    properties: Dict[str, any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

@dataclass
class ParsedCommand:
    action: ActionType
    target: Optional[str] = None
    indirect_object: Optional[str] = None
    preposition: Optional[str] = None
    confidence: float = 0.0
    original_input: str = ""
    valid_in_context: bool = True


class EnhancedNLPParser:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.DIR_CANON = {
            "north":"n","n":"n",
            "south":"s","s":"s",
            "east":"e","e":"e",
            "west":"w","w":"w",
            "up":"u","u":"u",
            "down":"d","d":"d",
            "northeast":"ne","ne":"ne",
            "northwest":"nw","nw":"nw",
            "southeast":"se","se":"se",
            "southwest":"sw","sw":"sw",
            "in":"in","into":"in",
            "out":"out",
        }
    
        PREPOSITIONS = ['in','on','with','to','from','at','into','onto','under','over','through']
        dir_alts = "|".join(sorted(map(re.escape, self.DIR_CANON.keys()), key=len, reverse=True))
        self.DIR_RE = re.compile(rf"\b({dir_alts})\b", re.IGNORECASE)

        self.object_knowledge = {} 
        self._initialize_affordances()
        self.action_object_compatibility = {}
        self.valid_actions = set()

        self.movement_verbs = ["go", "walk", "head", "enter", "leave", "climb"]
        self.verb_mappings = {
            ActionType.MOVEMENT: self.movement_verbs,

            ActionType.SAVE: ["save", "save game", "save my game"],
            ActionType.RESTORE: ["restore", "restore game", "restore my last save"],
            ActionType.RESTART: ["restart"],
            ActionType.VERBOSE: ["give full descriptions", "full descriptions", "verbose"],
            ActionType.SUPERBRIEF: ["don't describe the area", "dont describe the area", "superbrief"],
            ActionType.BRIEF: ["describe the area", "brief"],
            ActionType.SCORE: ["what is my score", "score"],
            ActionType.DIAGNOSE: ["how am i doing", "diagnose"],
            ActionType.QUIT: ["quit the game", "quit"],
            ActionType.WAIT: ["wait", "z"],
            ActionType.AGAIN: ["again", "g"],
            ActionType.INVENTORY: ["what am i carrying", "inventory", "i"],
            ActionType.LOOK: ["look around", "look", "l"],

            ActionType.TAKE: ["pick up", "take", "get", "grab", "acquire", "collect", "gather", "snatch"],
            ActionType.DROP: ["put down", "drop", "discard", "release", "abandon", "leave"],
            ActionType.THROW: ["throw", "toss", "hurl", "lob"],
            ActionType.OPEN: ["open", "unseal", "unfasten", "unbolt"],
            ActionType.CLOSE: ["close", "shut", "seal", "lock", "fasten", "bolt"],
            ActionType.UNLOCK: ["unlock"],
            ActionType.EXAMINE: ["look at", "examine", "inspect", "check", "study", "observe", "analyze", "x"],
            ActionType.ATTACK: ["attack", "hit", "strike", "fight", "kill", "assault", "bash", "kick"],
            ActionType.USE: ["use", "operate", "activate", "apply"],
            ActionType.EAT: ["eat", "consume", "devour", "taste", "swallow"],
            ActionType.DRINK: ["drink", "sip", "gulp", "quaff"],
            ActionType.LIGHT: ["light", "ignite", "burn", "kindle", "illuminate"],
            ActionType.READ: ["read", "peruse", "scan", "review"],
            ActionType.GIVE: ["give", "offer", "hand", "present", "donate", "bestow"],
            ActionType.PUT: ["put", "place", "insert", "position", "set", "store"],
            ActionType.TALK: ["talk", "speak", "say", "tell", "ask", "chat", "converse"],
        }
        # Precompute embeddings for action clusters
        self.action_embeddings = {}
        for action_type, keywords in self.verb_mappings.items():
            # Create representative embedding for each action type
            action_text = " ".join(keywords[:3])  # first 3 keywords
            self.action_embeddings[action_type] = self.model.encode(action_text)
            
    def _find_direction(self, text: str) -> Optional[str]:
        m = self.DIR_RE.search(text)
        if not m:
            return None
        tok = m.group(1).lower()
        return self.DIR_CANON.get(tok)
    
    def _initialize_affordances(self):
        """
        Initialize object affordances based on Fulda et al. (2017)
        Using word embeddings to determine what actions can be performed with objects
        """
        # Common game object affordances
        self.base_affordances = {
            # Containers
            "mailbox": {"open", "close", "examine", "look in", "put in"},
            "box": {"open", "close", "examine", "take", "put in"},
            "chest": {"open", "close", "examine", "unlock", "put in"},
            "bag": {"open", "close", "take", "examine", "put in"},
            "sack": {"open", "close", "take", "examine", "put in"},
            
            # Readable items
            "leaflet": {"read", "take", "drop", "examine"},
            "book": {"read", "take", "drop", "examine", "open"},
            "note": {"read", "take", "drop", "examine"},
            "letter": {"read", "take", "drop", "examine"},
            "scroll": {"read", "take", "drop", "examine", "unroll"},
            
            # Light sources
            "lamp": {"light", "extinguish", "take", "drop", "examine"},
            "lantern": {"light", "extinguish", "take", "drop", "examine"},
            "torch": {"light", "extinguish", "take", "drop", "examine"},
            "candle": {"light", "extinguish", "take", "drop", "examine"},
            
            # Weapons
            "sword": {"take", "drop", "examine", "attack with", "wield"},
            "knife": {"take", "drop", "examine", "attack with", "cut with"},
            "club": {"take", "drop", "examine", "attack with", "wield"},
            
            # Food/Drink
            "bread": {"eat", "take", "drop", "examine"},
            "water": {"drink", "take", "drop", "examine"},
            "bottle": {"open", "drink", "take", "drop", "examine"},
            
            # Doors/Passages
            "door": {"open", "close", "unlock", "examine", "go through"},
            "trapdoor": {"open", "close", "examine", "go down"},
            "window": {"open", "close", "examine", "look through", "break"},
            
            # NPCs/Creatures
            "troll": {"attack", "talk to", "give to", "examine"},
            "thief": {"attack", "talk to", "give to", "examine"},
        }
        
        # embeddings for affordance inference
        self.affordance_embeddings = {}
        for obj, affordances in self.base_affordances.items():
            self.affordance_embeddings[obj] = self.model.encode(obj)
    
    def infer_affordances(self, object_name: str) -> Set[str]:
        """
        Infer affordances for an object using embeddings (Fulda et al. 2017)
        find similar objects and use their affordances if not seen before
        """
        object_lower = object_name.lower()
        
        # Check explicit affordances
        if object_lower in self.base_affordances:
            return self.base_affordances[object_lower]
        
        # Use embedding similarity to find similar objects
        obj_embedding = self.model.encode(object_lower)
        best_match = None
        best_similarity = 0.0
        
        for known_obj, known_embedding in self.affordance_embeddings.items():
            similarity = cosine_similarity([obj_embedding], [known_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = known_obj
        
        # Similar object (threshold 0.6), use its affordances
        if best_similarity > 0.6 and best_match:
            return self.base_affordances[best_match].copy()
        
        # Default affordances for unknown objects
        return {"examine", "take", "drop"}

    def parse_command(self, user_input: str, context: Optional[Dict] = None) -> ParsedCommand:
        original = user_input
        user_input = self._normalize_input(user_input)
        action_type = self._extract_action(user_input)
        target, preposition, indirect = self._extract_entities(user_input, action_type)
        valid = self._is_action_valid(action_type, target)
        confidence = self._calculate_confidence(user_input, action_type, valid)
        
        return ParsedCommand(
            action=action_type,
            target=target,
            indirect_object=indirect,
            preposition=preposition,
            confidence=confidence,
            original_input=original,
            valid_in_context=valid
        )

    def _extract_movement_direction(self, text: str) -> Optional[str]:
        dirs = [
            "into", "in", "out",
            "n","s","e","w","u","d","ne","nw","se","sw",
            "north","south","east","west","up","down",
            "northeast","northwest","southeast","southwest",
        ]
        m = re.search(r"\b(" + "|".join(map(re.escape, dirs)) + r")\b", text)
        if not m:
            return None
        d = m.group(1)
        return "in" if d == "into" else d
    
    def _canon_noun(self, s: Optional[str]) -> Optional[str]:
        if not s: return s
        s = re.sub(r"\b(my|your|his|her|their|our)\b\s*", "", s)
        s = re.sub(r"\b(the|a|an)\b\s*", "", s)
        s = re.sub(r"'s\b.*$", "", s)  # "goblin's head" -> "goblin"
        return s.strip() or None
    
    def _normalize_input(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s']+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    def _extract_action(self, user_input: str) -> ActionType:
        text = user_input  # already normalized

        # 1) Phrase-first (longest first) across verb_mappings (movement verbs only; no dir tokens)
        pairs = []
        for a, kws in self.verb_mappings.items():
            for kw in kws:
                pairs.append((a, kw))
        pairs.sort(key=lambda p: len(p[1]), reverse=True)

        for action_type, kw in pairs:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return action_type

        # 2) Bare-direction fallback ONLY (single-token direction like "s", "south", "out")
        if self._find_direction(text) and len(text.split()) == 1:
            return ActionType.MOVEMENT

        # 3) Embedding fallback (do not choose MOVEMENT here)
        input_embedding = self.model.encode(text)
        best_action, best_sim = ActionType.UNKNOWN, 0.0
        for a, emb in self.action_embeddings.items():
            sim = cosine_similarity([input_embedding], [emb])[0][0]
            if sim > best_sim:
                best_action, best_sim = a, sim
        return best_action if (best_sim > 0.55 and best_action != ActionType.MOVEMENT) else ActionType.UNKNOWN


    def _extract_entities(self, user_input: str, action: ActionType):
        text = user_input

        # MOVEMENT: return early for directions
        if action == ActionType.MOVEMENT:
            d = self._find_direction(text)
            return d, None, None

        # Remove multi-word action phrases for THIS action
        multi = [kw for kw in self.verb_mappings.get(action, []) if " " in kw]
        for phrase in sorted(multi, key=len, reverse=True):
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)

        # Tokenize
        tokens = [t for t in text.split() if t]

        # Remove single-word action verbs for THIS action
        action_words = set(kw for kw in self.verb_mappings.get(action, []) if " " not in kw)

        # Common filler words to drop
        filler_words = {
            'i', 'want', 'can', 'you', 'please', 'would', 'like', 'could', 'should',
            'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'their', 'our',
            'is', 'am', 'are', 'the', 'a', 'an'
        }

        remaining = [t for t in tokens if not (t in action_words or t in filler_words)]

        PREPOSITIONS = ['in', 'on', 'with', 'to', 'from', 'at', 'into', 'onto', 'under', 'over', 'through']

        # preposition position
        prep, prep_idx = None, -1
        for i, tok in enumerate(remaining):
            if tok in PREPOSITIONS:
                prep, prep_idx = tok, i
                break

        target = indirect = None
        if prep_idx > -1:
            if prep_idx > 0:
                target = ' '.join(remaining[:prep_idx])
            if prep_idx < len(remaining) - 1:
                indirect = ' '.join(remaining[prep_idx + 1:])
        elif remaining:
            target = ' '.join(remaining)

        # Return without reference resolution
        return target, prep, indirect
    
    def _is_action_valid(self, action: ActionType, target: Optional[str]) -> bool:
        # Reject unknown actions
        if action == ActionType.UNKNOWN:
            return False

        actions_requiring_target = {
            ActionType.TAKE, ActionType.DROP, ActionType.OPEN, ActionType.CLOSE,
            ActionType.ATTACK, ActionType.READ, ActionType.EAT, ActionType.DRINK,
            ActionType.LIGHT, ActionType.PUT, ActionType.GIVE, ActionType.UNLOCK,
            ActionType.THROW, ActionType.EXAMINE, ActionType.USE
        }
        if action in actions_requiring_target and not target:
            return False

        return True
    
    def _calculate_confidence(self, user_input: str, action: ActionType, valid: bool) -> float:
        """
        Calculate confidence score based on multiple factors
        """
        base_confidence = 0.5
        
        # Boost for valid actions
        if valid:
            base_confidence += 0.2
        
        # Boost for exact keyword matches
        input_lower = user_input.lower()
        if action in self.verb_mappings:
            for keyword in self.verb_mappings[action]:
                if keyword in input_lower:
                    base_confidence += 0.2
                    break
        
        # Penalty for unknown action
        if action == ActionType.UNKNOWN:
            base_confidence = 0.1
        
        return min(base_confidence, 1.0)
    
    def _canon_prep(self, p: Optional[str]) -> Optional[str]:
        if not p: return p
        return "in" if p == "into" else p
    
    def to_zork_command(self, parsed: ParsedCommand) -> str:
        action = parsed.action
        target = self._canon_noun(parsed.target)
        indirect = self._canon_noun(parsed.indirect_object)
        prep = self._canon_prep(parsed.preposition)

        if action == ActionType.MOVEMENT:
            if target:
                direction_map = {
                    'north':'n','south':'s','east':'e','west':'w',
                    'up':'u','down':'d','northeast':'ne','northwest':'nw',
                    'southeast':'se','southwest':'sw',
                    'n':'n','s':'s','e':'e','w':'w','u':'u','d':'d',
                    'ne':'ne','nw':'nw','se':'se','sw':'sw','in':'in','out':'out'
                }
                return direction_map.get(target, target)
            return "go"

        meta = {
            ActionType.SAVE:"save", ActionType.RESTORE:"restore", ActionType.RESTART:"restart",
            ActionType.VERBOSE:"verbose", ActionType.BRIEF:"brief", ActionType.SUPERBRIEF:"superbrief",
            ActionType.SCORE:"score", ActionType.DIAGNOSE:"diagnose", ActionType.QUIT:"quit",
            ActionType.WAIT:"wait", ActionType.AGAIN:"again", ActionType.INVENTORY:"i", ActionType.LOOK:"look"
        }
        if action in meta:
            return meta[action]

        if action == ActionType.THROW:
            if target and prep == "at" and indirect:
                return f"throw {target} at {indirect}"
            return f"throw {target}".strip()

        if action == ActionType.PUT:
            if target and prep and indirect:
                return f"put {target} {prep} {indirect}"
            return f"put {target}".strip()

        if action == ActionType.TAKE:
            if target and prep == "from" and indirect:
                return f"take {target} from {indirect}"
            return f"take {target}".strip()

        if action == ActionType.GIVE:
            if target and (prep == "to" or indirect):
                return f"give {target} to {indirect}".strip()
            
        if action == ActionType.UNLOCK:
            if target and prep == "with" and indirect:
                return f"unlock {target} with {indirect}"
            return f"unlock {target}".strip()

        action_map = {
            ActionType.DROP:"drop", ActionType.OPEN:"open", ActionType.CLOSE:"close",
            ActionType.EXAMINE:"examine", ActionType.READ:"read", ActionType.ATTACK:"attack",
            ActionType.EAT:"eat", ActionType.DRINK:"drink", ActionType.LIGHT:"light",
            ActionType.TALK:"talk to", ActionType.EXTINGUISH:"extinguish", ActionType.USE:"use"
        }
        if action in action_map:
            base = action_map[action]
            return f"{base} {target}".strip()

        return parsed.original_input


if __name__ == "__main__":
    parser = EnhancedNLPParser()
    
    # --- Test cases: (input_text, expected_zork_command) ---
    # --- Test cases with true Zork-style expected commands ---
    # Each expected is a set of acceptable canonical outputs after simple normalization.

    test_cases = [
        ("south",                         {"SOUTH", "S"}),
        ("s",                             {"S", "SOUTH"}),
        ("go south",                      {"SOUTH", "S"}),
        ("walk s",                        {"SOUTH", "S"}),
        ("go north",                      {"NORTH", "N"}),
        ("go east",                       {"EAST", "E"}),
        ("go west",                       {"WEST", "W"}),
        ("northwest",                     {"NORTHWEST", "NW"}),
        ("go NE",                         {"NE"}),
        ("climb down",                    {"DOWN", "D"}),
        ("go out",                        {"OUT"}),
        ("go into the cave",              {"IN"}),
        ("go in the window",              {"IN", "ENTER WINDOW", "GO THROUGH WINDOW"}),

        ("look",                          {"LOOK", "L"}),
        ("l",                             {"LOOK", "L"}),
        ("look around",                   {"LOOK", "L"}),
        ("LOOK!!!",                       {"LOOK", "L"}),
        ("inventory",                     {"INVENTORY", "I"}),
        ("what am i carrying?",           {"INVENTORY", "I"}),

        ("examine mailbox",               {"EXAMINE MAILBOX", "X MAILBOX"}),
        ("x mailbox",                     {"EXAMINE MAILBOX", "X MAILBOX"}),
        ("look at the mailbox",           {"EXAMINE MAILBOX", "X MAILBOX"}),

        ("take lamp",                      {"TAKE LAMP"}),
        ("get lamp",                       {"TAKE LAMP"}),
        ("drop lamp",                      {"DROP LAMP"}),
        ("put lamp in bag",                {"PUT LAMP IN BAG"}),
        ("take lamp from bag",             {"TAKE LAMP FROM BAG"}),
        ("open door",                      {"OPEN DOOR"}),
        ("close door",                     {"CLOSE DOOR"}),
        ("unlock door with key",           {"UNLOCK DOOR WITH KEY"}),

        ("light the lamp",                 {"LIGHT LAMP"}),
        ("extinguish lamp",                {"EXTINGUISH LAMP"}),
        ("read leaflet",                   {"READ LEAFLET"}),

        ("throw sword at troll",           {"THROW SWORD AT TROLL"}),
        ("attack troll",                   {"ATTACK TROLL"}),
        ("attack troll with sword",        {"ATTACK TROLL WITH SWORD", "ATTACK TROLL"}),
        ("kill troll with sword",          {"ATTACK TROLL WITH SWORD", "ATTACK TROLL"}),
        ("give coin to troll",             {"GIVE COIN TO TROLL"}),

        # Meta
        ("save",                           {"SAVE"}),
        ("save my game",                   {"SAVE"}),
        ("restore",                        {"RESTORE"}),
        ("restore my last save",           {"RESTORE"}),
        ("restart",                        {"RESTART"}),
        ("restart the game",               {"RESTART"}),
        ("verbose",                        {"VERBOSE"}),
        ("give full descriptions",         {"VERBOSE"}),
        ("brief",                          {"BRIEF"}),
        ("describe the area",              {"BRIEF"}),
        ("superbrief",                     {"SUPERBRIEF"}),
        ("don't describe the area",        {"SUPERBRIEF"}),
        ("score",                          {"SCORE"}),
        ("what is my score?",              {"SCORE"}),
        ("diagnose",                       {"DIAGNOSE"}),
        ("How am I doing?",                {"DIAGNOSE"}),
        ("wait",                           {"WAIT", "Z"}),
        ("z",                              {"WAIT", "Z"}),
        ("again",                          {"AGAIN", "G"}),
        ("g",                              {"AGAIN", "G"}),
        ("quit the game",                  {"QUIT"}),
        ("quit",                           {"QUIT"}),

        ("open my bag",                    {"OPEN BAG", "OPEN SACK"}),
        ("open the sack",                  {"OPEN SACK", "OPEN BAG"}),

        # whitespace / case noise 
        ("  go   south   ",                {"SOUTH", "S"}),
        ("Go South",                       {"SOUTH", "S"}),
        ("GO SOUTH",                       {"SOUTH", "S"}),
    ]

    import time
    from typing import Optional

    def canonicalize(cmd: Optional[str]) -> str: 
        """Uppercase, strip, collapse spaces, remove trailing punctuation.""" 
        if not cmd:
            return "" 
        import re 
        cmd = cmd.upper().strip() 
        cmd = re.sub(r"[!?.,;:]+$", "", cmd) 
        cmd = re.sub(r"\s+", " ", cmd) 
        return cmd 

    total = len(test_cases) 
    passes = 0 
    failures = [] 

    for i, (inp, expected_set) in enumerate(test_cases, 1): 
        parsed = parser.parse_command(inp) 
        zork_cmd = parser.to_zork_command(parsed) 
        
        got_canon = canonicalize(zork_cmd) 
        exp_canon = {canonicalize(e) for e in expected_set} 
        ok = got_canon in exp_canon 
        passes += int(ok) 
        
        status = "PASS" if ok else "FAIL" 
        print(f"\n[{i:02d}] {status} Input: {inp!r}") 
        print(f" Expected any of: {sorted(exp_canon)}") 
        print(f" Got: {got_canon!r}") 
        
        action_val = getattr(getattr(parsed, "action", None), "value", None) 
        target = getattr(parsed, "target", None) 
        indirect = getattr(parsed, "indirect_object", None) 
        valid_in_context = getattr(parsed, "valid_in_context", None) 
        confidence = getattr(parsed, "confidence", None) 
        if action_val is not None: 
            print(f" Action: {action_val}") 
        if target: 
            print(f" Target: {target}") 
        if indirect: 
            print(f" Indirect: {indirect}") 
        if valid_in_context is not None: 
            print(f" Valid in context: {valid_in_context}") 
        if confidence is not None: 
            try:
                print(f" Confidence: {float(confidence):.2f}") 
            except Exception: 
                print(f" Confidence: {confidence}") # Affordances
        if target and hasattr(parser, "infer_affordances"): 
            try: 
                aff = parser.infer_affordances(target) 
                print(f" {target} affordances: {aff}") 
            except Exception: 
                pass 

        if not ok: 
            failures.append({ 
                "index": i, 
                "input": inp, 
                "expected_any": sorted(exp_canon), 
                "got": got_canon, 
                }) 
        
        print(f"Passed {passes}/{total} ({(passes/total)*100:.1f}%)") 
        if failures: 
            print("Failures:") 
            for f in failures: 
                print(f" [{f['index']:02d}] input={f['input']!r} expected_any={f['expected_any']} got={f['got']!r}")