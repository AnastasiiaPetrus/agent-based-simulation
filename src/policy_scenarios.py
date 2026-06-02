import random


POLICY_INTENSITY_VALUES = {"Low": 0.2, "Medium": 0.4, "High": 0.7}


POLICY_SCENARIO_LIBRARY = {
    "Targeted support for high-risk children": {
        "Low": [
            {
                "name": "Optional mentoring offer",
                "description": (
                    "The person is offered access to a mentor or advisor who can meet occasionally to "
                    "discuss goals, challenges, planning, relationships, or future options. Participation "
                    "is voluntary, and declining has no negative consequence."
                ),
            },
            {
                "name": "Voluntary skill-building access",
                "description": (
                    "The person is offered free access to tutoring, coaching, training, or skill-building "
                    "support. The person may choose whether to use the service, and non-participation does "
                    "not affect their ordinary status."
                ),
            },
            {
                "name": "Open community activities",
                "description": (
                    "The person is invited to join structured activities such as sports, arts, clubs, "
                    "workshops, volunteering, or community programs. Attendance is optional and not linked "
                    "to monitoring or compliance decisions."
                ),
            },
            {
                "name": "Optional practical assistance",
                "description": (
                    "The person or household is offered practical help with communication, routines, "
                    "applications, transport, appointments, or access to available services. The offer can "
                    "be accepted or declined without penalty."
                ),
            },
        ],
        "Medium": [
            {
                "name": "Supportive case coordination",
                "description": (
                    "A support worker or case coordinator is assigned to help the person identify needs, "
                    "navigate services, and coordinate communication with relevant organizations. "
                    "Participation remains voluntary and assistance-oriented."
                ),
            },
            {
                "name": "Collaborative support plan",
                "description": (
                    "The person and support provider jointly create a structured plan that may include "
                    "coaching, mentoring, counseling, skill-building, goal-setting, or practical assistance. "
                    "The plan is reviewed periodically, but non-participation does not trigger sanctions."
                ),
            },
            {
                "name": "Voluntary wellbeing referral",
                "description": (
                    "The person is offered access to counseling, psychological support, health support, or "
                    "wellbeing services. Services begin only with appropriate consent and are not tied to "
                    "punishment, surveillance, or mandatory compliance."
                ),
            },
            {
                "name": "Engagement support plan",
                "description": (
                    "A support provider and the person develop a plan to improve participation in ordinary "
                    "life domains such as education, work, community activities, care responsibilities, or "
                    "service access. The plan focuses on removing barriers and expanding options."
                ),
            },
        ],
        "High": [
            {
                "name": "Long-term wraparound support",
                "description": (
                    "The person receives sustained, coordinated support across multiple domains such as "
                    "housing, health, income, education, employment, family, community, and legal or "
                    "administrative needs. The approach is intensive but remains focused on assistance "
                    "rather than control."
                ),
            },
            {
                "name": "Frequent supportive check-ins",
                "description": (
                    "A support worker meets regularly with the person to identify problems early, offer "
                    "practical help, and adjust services. The meetings are framed around support and "
                    "problem-solving, not compliance monitoring."
                ),
            },
            {
                "name": "Priority access to stabilizing services",
                "description": (
                    "The person receives faster access to services such as housing assistance, food "
                    "support, transport, healthcare, counseling, education, employment support, financial "
                    "advice, or safe community programs. Access is based on need rather than punishment or "
                    "restriction."
                ),
            },
            {
                "name": "Long-term opportunity pathway",
                "description": (
                    "The person receives structured help accessing training, employment, internships, "
                    "apprenticeships, education, grants, community roles, or other future-oriented "
                    "opportunities. The policy aims to expand available options rather than control "
                    "behavior."
                ),
            },
        ],
    },
    "Surveillance of high-risk children": {
        "Low": [
            {
                "name": "Basic participation monitoring",
                "description": (
                    "An organization regularly records whether the person attends scheduled appointments, "
                    "programs, services, work, education, or other relevant activities. No new support or "
                    "restriction is automatically provided."
                ),
            },
            {
                "name": "Periodic record review",
                "description": (
                    "Staff periodically review existing records, reports, complaints, incidents, missed "
                    "appointments, or prior interactions involving the person. The review is informational "
                    "and does not itself impose new requirements."
                ),
            },
            {
                "name": "Passive status review",
                "description": (
                    "The person's status is reviewed using existing administrative or organizational "
                    "records. No active contact occurs unless new information is added to the file."
                ),
            },
            {
                "name": "Limited internal flag",
                "description": (
                    "A small number of authorized staff can see that the person is assigned to an internal "
                    "monitoring category. The flag is used for awareness and review, not for providing "
                    "support or imposing restrictions by itself."
                ),
            },
        ],
        "Medium": [
            {
                "name": "Scheduled monitoring meetings",
                "description": (
                    "The person is required to attend periodic meetings with an official, staff member, or "
                    "monitoring body to review attendance, conduct, compliance indicators, or recent events. "
                    "The meetings primarily collect and update information."
                ),
            },
            {
                "name": "Cross-organization information sharing",
                "description": (
                    "Multiple organizations or agencies exchange information about the person's attendance, "
                    "reported incidents, complaints, contacts, or other recorded concerns. The main "
                    "function is institutional awareness and coordinated review."
                ),
            },
            {
                "name": "Trigger-based escalation review",
                "description": (
                    "Specific recorded events, such as repeated missed appointments, reports, complaints, "
                    "conflicts, or incidents, automatically trigger a higher-level review of the person's "
                    "status."
                ),
            },
            {
                "name": "Context and association recording",
                "description": (
                    "Institutions record repeated contexts connected to reports or incidents, such as "
                    "locations, events, groups, online spaces, or times of day. The policy tracks patterns "
                    "but does not itself restrict movement, contact, or participation."
                ),
            },
        ],
        "High": [
            {
                "name": "Frequent compulsory reporting for monitoring",
                "description": (
                    "The person must report frequently to a designated monitoring authority so officials "
                    "can review conduct, attendance, location, contacts, or recent events. The primary "
                    "purpose is observation and documentation."
                ),
            },
            {
                "name": "Home, workplace, program, or field checks",
                "description": (
                    "Monitoring staff may conduct checks at relevant locations, such as the person's home, "
                    "workplace, service site, program location, or other places connected to the review. "
                    "These checks focus on verification rather than direct service provision."
                ),
            },
            {
                "name": "Multi-agency watchlist status",
                "description": (
                    "The person is assigned a formal watchlist or high-monitoring status visible to "
                    "multiple organizations or authorities. The status increases institutional attention "
                    "and the likelihood of further review."
                ),
            },
            {
                "name": "Rapid monitoring escalation protocol",
                "description": (
                    "Minor incidents, missed meetings, complaints, reports, or new records quickly lead to "
                    "more intensive monitoring, broader information sharing, or referral to a stricter "
                    "review process."
                ),
            },
        ],
    },
    "Coercive preventive intervention for high-risk children": {
        "Low": [
            {
                "name": "Mandatory counseling or behavior sessions",
                "description": (
                    "The person is required to attend counseling, coaching, behavioral sessions, or "
                    "prevention meetings regardless of voluntary agreement. Non-attendance may trigger "
                    "consequences."
                ),
            },
            {
                "name": "Formal conduct agreement",
                "description": (
                    "The person must follow a written agreement that sets rules, obligations, and "
                    "consequences for violations. The agreement is imposed as a preventive measure."
                ),
            },
            {
                "name": "Compulsory attendance or participation plan",
                "description": (
                    "An organization imposes a mandatory plan requiring attendance, check-ins, documented "
                    "arrival, or regular reporting. Failure to comply may lead to formal consequences."
                ),
            },
            {
                "name": "Restricted activity access after repeated indicators",
                "description": (
                    "After repeated recorded warning signs, the person is barred from specific activities, "
                    "events, platforms, services, or settings linked to prior concerns. The restriction is "
                    "mandatory and preventive."
                ),
            },
        ],
        "Medium": [
            {
                "name": "Compulsory structured program",
                "description": (
                    "The person must attend a structured program during specified hours. The program is "
                    "mandatory and replaces part of the person's ordinary schedule or free time."
                ),
            },
            {
                "name": "Mandatory reporting to an authority",
                "description": (
                    "The person must regularly report to an assigned officer, official, or authority. "
                    "Failure to report may result in sanctions or stricter intervention."
                ),
            },
            {
                "name": "Place-based movement restrictions",
                "description": (
                    "The person is prohibited from entering certain places, attending certain events, "
                    "contacting certain groups, or being in specified areas during certain times. The "
                    "restriction is imposed before a new harmful event or violation occurs."
                ),
            },
            {
                "name": "Mandated placement or setting change",
                "description": (
                    "The person is moved to another program, service setting, residential arrangement, "
                    "workplace setting, educational setting, or supervised environment without full "
                    "voluntary consent. The change is justified as preventive control."
                ),
            },
        ],
        "High": [
            {
                "name": "Electronic location monitoring",
                "description": (
                    "The person is required to wear or carry a location-monitoring device. Authorities use "
                    "the device to enforce preventive rules or movement limits."
                ),
            },
            {
                "name": "Strict curfew order",
                "description": (
                    "The person is prohibited from leaving a residence or designated location during "
                    "specified hours. Violations may trigger sanctions or stricter restrictions."
                ),
            },
            {
                "name": "Residential preventive placement",
                "description": (
                    "The person is placed in a structured residential program or supervised setting with "
                    "continuous oversight. The placement substantially limits ordinary movement, privacy, "
                    "and daily autonomy."
                ),
            },
            {
                "name": "Preemptive custody-like restriction",
                "description": (
                    "In a fictional or hypothetical setting, the person is placed under severe preventive "
                    "restriction without a prior completed violation or offense. The measure resembles "
                    "custody, detention, or confinement and is justified solely by predicted future risk."
                ),
            },
        ],
    },
}


def choose_policy_scenario(policy, intensity_tier):
    policy_entry = POLICY_SCENARIO_LIBRARY.get(policy)
    if not policy_entry:
        return None

    measures = policy_entry.get(intensity_tier)
    if not measures:
        return None

    selected_measure = random.choice(measures)
    return {
        "policy": policy,
        "measure": selected_measure["name"],
        "description": selected_measure["description"],
        "applies_to": "flagged agents",
    }


def policy_intensity_value(intensity_tier):
    return POLICY_INTENSITY_VALUES.get(intensity_tier, POLICY_INTENSITY_VALUES["Medium"])
