import face_recognition
import numpy as np


def get_face_descriptor(image_np):
    """
    Used for ADDING cases. We keep the intensive scan here to ensure
    we get a high-quality descriptor for the database.
    """
    # 1. Try a fast, standard scan first
    face_locations = face_recognition.face_locations(image_np)

    # 2. If no faces are found, try a more intensive upsampled scan
    if len(face_locations) == 0:
        face_locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=2)

    if len(face_locations) == 0:
        return None

    if len(face_locations) > 1:
        return 'multiple'

    # If we get here, there is exactly one face
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    return face_encodings[0]


def find_matches_in_image(image_np, known_descriptors, known_names, tolerance=0.55):
    """
    Used for LIVE scanning. Optimized for speed.
    REMOVED the intensive upsampling fallback to prevent lag.
    """
    # 1. Fast scan only.
    # model='hog' is faster than 'cnn'.
    unknown_face_locations = face_recognition.face_locations(image_np, model='hog')

    # If no faces found, return immediately (Don't waste time upsampling)
    if len(unknown_face_locations) == 0:
        return {
            'matches_found': 0,
            'match_results': []
        }

    unknown_face_encodings = face_recognition.face_encodings(image_np, unknown_face_locations)

    match_results = []
    unique_matched_names = set()

    for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
        matches = face_recognition.compare_faces(known_descriptors, face_encoding, tolerance)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(known_descriptors, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                unique_matched_names.add(name)

        match_results.append({
            'name': name,
            'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left}
        })

    matches_found = len(unique_matched_names)

    return {
        'matches_found': matches_found,
        'match_results': match_results
    }