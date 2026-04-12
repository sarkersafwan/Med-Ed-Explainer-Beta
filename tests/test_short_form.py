from tools.analyze import _build_scene_briefs
from tools.generate_script import _allocate_scene_word_targets
from tools.models import MedicalContent, SceneBrief


def test_short_form_scene_briefs_stay_within_requested_duration():
    content = MedicalContent(
        topic="Cyanide Poisoning",
        clinical_vignette="A factory worker suddenly becomes confused, tachypneic, and collapses after smoke inhalation.",
        question_stem="Which mechanism best explains this patient's rapid deterioration?",
        pathophysiology=(
            "Cyanide binds ferric iron in cytochrome c oxidase. "
            "That halts oxidative phosphorylation and prevents cells from using oxygen. "
            "ATP production crashes, forcing anaerobic metabolism and lactic acidosis."
        ),
        educational_objective="Explain how cyanide stops aerobic metabolism within seconds.",
        bottom_line="Cyanide poisoning causes histotoxic hypoxia by blocking the electron transport chain.",
    )

    scenes = _build_scene_briefs(
        content,
        [
            "Cyanide binds cytochrome c oxidase.",
            "Oxidative phosphorylation stops.",
            "Cells switch to anaerobic metabolism and generate lactate.",
        ],
        [],
        0.25,
    )

    assert [scene.purpose for scene in scenes] == ["hook", "mechanism", "takeaway"]
    assert round(sum(scene.estimated_minutes for scene in scenes), 2) == 0.25
    assert all(scene.estimated_minutes > 0 for scene in scenes)


def test_short_form_word_targets_match_total_budget():
    scenes = [
        SceneBrief(
            scene_number=1,
            scene_title="The Patient",
            purpose="hook",
            key_content="Hook",
            estimated_minutes=0.06,
        ),
        SceneBrief(
            scene_number=2,
            scene_title="The Mechanism",
            purpose="mechanism",
            key_content="Mechanism",
            estimated_minutes=0.13,
        ),
        SceneBrief(
            scene_number=3,
            scene_title="The Takeaway",
            purpose="takeaway",
            key_content="Takeaway",
            estimated_minutes=0.06,
        ),
    ]

    targets = _allocate_scene_word_targets(scenes, total_target_words=38)

    assert sum(targets) == 38
    assert all(target >= 10 for target in targets)
    assert targets[1] > targets[0]
