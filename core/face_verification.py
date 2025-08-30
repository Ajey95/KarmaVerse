import cv2
import face_recognition
import numpy as np
import os
import pickle
import logging
from typing import List, Tuple, Dict, Optional
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FaceVerificationSystem:
    """
    Handles face recognition, verification against authorized faces, and semi-supervised learning
    for automatic identity clustering and new face detection.
    """
    
    def __init__(self, authorized_faces_dir: str = "data/authorized_faces/"):
        self.authorized_faces_dir = authorized_faces_dir
        self.authorized_encodings = []
        self.authorized_names = []
        self.face_index = None
        self.similarity_threshold = 0.6  # Adjust based on accuracy needs
        
        # Semi-supervised learning components
        self.unknown_faces_dir = os.path.join(authorized_faces_dir, "unknown_clusters")
        self.unknown_encodings = []
        self.unknown_metadata = []  # Store timestamp, location, etc.
        self.face_clusters = {}  # cluster_id -> list of face_encodings
        self.cluster_representatives = {}  # cluster_id -> representative_encoding
        self.clustering_threshold = 0.4  # Tighter threshold for clustering
        self.min_cluster_size = 3  # Minimum faces to form a cluster
        self.new_identity_buffer = []  # Buffer for potential new identities
        
        # Ensure directories exist
        os.makedirs(authorized_faces_dir, exist_ok=True)
        os.makedirs(self.unknown_faces_dir, exist_ok=True)
        
        # Load authorized faces and unknown clusters
        self.load_authorized_faces()
        self.load_unknown_clusters()
        
    def load_unknown_clusters(self):
        """Load previously discovered unknown face clusters."""
        clusters_file = os.path.join(self.unknown_faces_dir, "face_clusters.pkl")
        
        if os.path.exists(clusters_file):
            try:
                with open(clusters_file, 'rb') as f:
                    cluster_data = pickle.load(f)
                    self.face_clusters = cluster_data.get('clusters', {})
                    self.cluster_representatives = cluster_data.get('representatives', {})
                    self.unknown_encodings = cluster_data.get('unknown_encodings', [])
                    self.unknown_metadata = cluster_data.get('unknown_metadata', [])
                
                logger.info(f"Loaded {len(self.face_clusters)} unknown face clusters")
            except Exception as e:
                logger.error(f"Error loading unknown clusters: {e}")
                self.face_clusters = {}
                self.cluster_representatives = {}
        
    def save_unknown_clusters(self):
        """Save unknown face clusters to persistent storage."""
        clusters_file = os.path.join(self.unknown_faces_dir, "face_clusters.pkl")
        
        try:
            cluster_data = {
                'clusters': self.face_clusters,
                'representatives': self.cluster_representatives,
                'unknown_encodings': self.unknown_encodings,
                'unknown_metadata': self.unknown_metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(clusters_file, 'wb') as f:
                pickle.dump(cluster_data, f)
            
            logger.info(f"Saved {len(self.face_clusters)} face clusters")
        except Exception as e:
            logger.error(f"Error saving clusters: {e}")
    
    def detect_and_verify_faces(self, frame: np.ndarray, location: str = "Unknown", 
                               timestamp: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Enhanced face detection with semi-supervised learning for unknown identity discovery.
        
        Args:
            frame: Input video frame
            location: Location where frame was captured
            timestamp: Timestamp of frame capture
            
        Returns:
            Tuple of (processed_frame, face_detections_info)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        face_info = []
        processed_frame = frame.copy()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check against authorized faces first
            is_authorized, person_name, confidence = self._verify_face(face_encoding)
            
            if is_authorized:
                # Authorized person detected
                face_data = {
                    'location': (top, right, bottom, left),
                    'is_authorized': True,
                    'person_name': person_name,
                    'confidence': confidence,
                    'identity_type': 'known_authorized',
                    'cluster_id': None
                }
                
                # Draw green rectangle for authorized faces
                cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"{person_name} ✓", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                logger.info(f"Authorized person detected: {person_name} (confidence: {confidence:.2f})")
                
            else:
                # Unknown face - apply semi-supervised learning
                cluster_result = self._process_unknown_face(
                    face_encoding, location, timestamp, (top, right, bottom, left)
                )
                
                face_data = {
                    'location': (top, right, bottom, left),
                    'is_authorized': False,
                    'person_name': None,
                    'confidence': confidence,
                    'identity_type': cluster_result['identity_type'],
                    'cluster_id': cluster_result.get('cluster_id'),
                    'cluster_confidence': cluster_result.get('cluster_confidence', 0.0),
                    'is_new_identity': cluster_result.get('is_new_identity', False)
                }
                
                # Blur unauthorized face
                processed_frame = self._blur_face_region(processed_frame, (top, right, bottom, left))
                
                # Add visual indicator for different types of unknown faces
                if cluster_result['identity_type'] == 'recurring_unknown':
                    # Orange border for recurring unknown identities
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 165, 255), 2)
                    cv2.putText(processed_frame, f"ID-{cluster_result['cluster_id']}", 
                               (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                elif cluster_result['identity_type'] == 'potential_new':
                    # Yellow border for potential new identities
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(processed_frame, "New?", (left, bottom + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    # Red border for single unknown faces
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(processed_frame, "Unknown", (left, bottom + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                logger.info(f"Unknown face detected: {cluster_result['identity_type']}")
            
            face_info.append(face_data)
        
        # Periodically update clusters
        if len(self.new_identity_buffer) >= 10:  # Process every 10 new faces
            self._update_face_clusters()
        
        return processed_frame, face_info
    
    def _process_unknown_face(self, face_encoding: np.ndarray, location: str, 
                             timestamp: str, face_coords: Tuple) -> Dict:
        """
        Process unknown face using semi-supervised learning approach.
        
        Args:
            face_encoding: Face encoding vector
            location: Location where face was detected
            timestamp: Detection timestamp
            face_coords: Face bounding box coordinates
            
        Returns:
            Dictionary with cluster analysis results
        """
        # Check if this face matches any existing unknown clusters
        best_cluster_id, cluster_confidence = self._match_to_cluster(face_encoding)
        
        if best_cluster_id is not None and cluster_confidence > self.clustering_threshold:
            # Face matches existing cluster - update cluster
            self.face_clusters[best_cluster_id].append({
                'encoding': face_encoding,
                'location': location,
                'timestamp': timestamp,
                'coordinates': face_coords
            })
            
            # Update cluster representative (running average)
            self._update_cluster_representative(best_cluster_id, face_encoding)
            
            return {
                'identity_type': 'recurring_unknown',
                'cluster_id': best_cluster_id,
                'cluster_confidence': cluster_confidence,
                'cluster_size': len(self.face_clusters[best_cluster_id])
            }
        
        else:
            # New face - add to buffer for clustering
            face_metadata = {
                'encoding': face_encoding,
                'location': location,
                'timestamp': timestamp,
                'coordinates': face_coords
            }
            
            self.new_identity_buffer.append(face_metadata)
            self.unknown_encodings.append(face_encoding)
            self.unknown_metadata.append(face_metadata)
            
            # Check if this might be a new recurring identity
            if len(self.new_identity_buffer) >= self.min_cluster_size:
                potential_cluster = self._check_for_new_cluster(face_encoding)
                
                if potential_cluster:
                    return {
                        'identity_type': 'potential_new',
                        'is_new_identity': True,
                        'similar_faces_count': len(potential_cluster)
                    }
            
            return {
                'identity_type': 'single_unknown',
                'is_new_identity': False
            }
    
    def _match_to_cluster(self, face_encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match face encoding to existing clusters.
        
        Args:
            face_encoding: Face encoding to match
            
        Returns:
            Tuple of (best_cluster_id, confidence_score)
        """
        if not self.cluster_representatives:
            return None, 0.0
        
        best_cluster_id = None
        best_similarity = 0.0
        
        for cluster_id, representative in self.cluster_representatives.items():
            # Calculate cosine similarity
            similarity = cosine_similarity([face_encoding], [representative])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        
        return best_cluster_id, best_similarity
    
    def _check_for_new_cluster(self, face_encoding: np.ndarray) -> Optional[List]:
        """
        Check if recent faces in buffer could form a new cluster with current face.
        
        Args:
            face_encoding: Current face encoding
            
        Returns:
            List of similar faces if potential cluster found, None otherwise
        """
        if len(self.new_identity_buffer) < self.min_cluster_size:
            return None
        
        # Get recent faces from buffer
        recent_faces = self.new_identity_buffer[-20:]  # Last 20 faces
        similar_faces = []
        
        for face_data in recent_faces:
            similarity = cosine_similarity([face_encoding], [face_data['encoding']])[0][0]
            
            if similarity > self.clustering_threshold:
                similar_faces.append(face_data)
        
        if len(similar_faces) >= self.min_cluster_size - 1:  # -1 because current face will be added
            return similar_faces
        
        return None
    
    def _update_face_clusters(self):
        """
        Update face clusters using DBSCAN clustering on accumulated unknown faces.
        """
        if len(self.new_identity_buffer) < self.min_cluster_size:
            return
        
        # Extract encodings from buffer
        buffer_encodings = np.array([face['encoding'] for face in self.new_identity_buffer])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - self.clustering_threshold,  # Convert similarity to distance
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(buffer_encodings)
        
        # Process clusters
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get faces in this cluster
            cluster_mask = clustering.labels_ == label
            cluster_faces = [self.new_identity_buffer[i] for i in range(len(self.new_identity_buffer)) if cluster_mask[i]]
            
            if len(cluster_faces) >= self.min_cluster_size:
                # Create new cluster
                cluster_id = f"unknown_{len(self.face_clusters) + 1}_{datetime.now().strftime('%Y%m%d')}"
                
                self.face_clusters[cluster_id] = cluster_faces
                
                # Calculate representative encoding (centroid)
                cluster_encodings = np.array([face['encoding'] for face in cluster_faces])
                representative = np.mean(cluster_encodings, axis=0)
                self.cluster_representatives[cluster_id] = representative
                
                logger.info(f"Created new face cluster: {cluster_id} with {len(cluster_faces)} faces")
        
        # Clear buffer and save clusters
        self.new_identity_buffer = []
        self.save_unknown_clusters()
    
    def _update_cluster_representative(self, cluster_id: str, new_encoding: np.ndarray):
        """
        Update cluster representative using exponential moving average.
        
        Args:
            cluster_id: Cluster to update
            new_encoding: New face encoding to incorporate
        """
        if cluster_id not in self.cluster_representatives:
            self.cluster_representatives[cluster_id] = new_encoding
        else:
            # Exponential moving average (alpha = 0.1)
            alpha = 0.1
            current_rep = self.cluster_representatives[cluster_id]
            self.cluster_representatives[cluster_id] = (1 - alpha) * current_rep + alpha * new_encoding
    
    def get_face_match_summary(self, face_info_list: List[Dict]) -> Dict:
        """
        Enhanced summary including semi-supervised learning insights.
        
        Args:
            face_info_list: List of face detection info
            
        Returns:
            Enhanced summary dictionary
        """
        total_faces = len(face_info_list)
        authorized_faces = sum(1 for face in face_info_list if face['is_authorized'])
        unauthorized_faces = total_faces - authorized_faces
        
        # Count different types of unknown faces
        recurring_unknown = sum(1 for face in face_info_list 
                               if face.get('identity_type') == 'recurring_unknown')
        potential_new = sum(1 for face in face_info_list 
                           if face.get('identity_type') == 'potential_new')
        single_unknown = sum(1 for face in face_info_list 
                            if face.get('identity_type') == 'single_unknown')
        
        authorized_names = [face['person_name'] for face in face_info_list 
                          if face['is_authorized'] and face['person_name']]
        
        # Get cluster information
        cluster_ids = [face.get('cluster_id') for face in face_info_list 
                      if face.get('cluster_id') is not None]
        unique_clusters = len(set(cluster_ids))
        
        return {
            'total_faces': total_faces,
            'authorized_faces': authorized_faces,
            'unauthorized_faces': unauthorized_faces,
            'authorized_names': authorized_names,
            'has_unauthorized': unauthorized_faces > 0,
            'semi_supervised_insights': {
                'recurring_unknown_identities': recurring_unknown,
                'potential_new_identities': potential_new,
                'single_unknown_faces': single_unknown,
                'active_unknown_clusters': unique_clusters,
                'total_discovered_clusters': len(self.face_clusters),
                'learning_buffer_size': len(self.new_identity_buffer)
            }
        }
    
    def get_clustering_report(self) -> Dict:
        """
        Generate comprehensive report on semi-supervised learning progress.
        
        Returns:
            Detailed clustering analysis report
        """
        report = {
            'summary': {
                'total_authorized_identities': len(self.authorized_names),
                'total_unknown_clusters': len(self.face_clusters),
                'total_unknown_faces_processed': len(self.unknown_encodings),
                'faces_in_learning_buffer': len(self.new_identity_buffer)
            },
            'cluster_details': {},
            'recommendations': []
        }
        
        # Analyze each cluster
        for cluster_id, faces in self.face_clusters.items():
            cluster_locations = [face['location'] for face in faces]
            location_counts = Counter(cluster_locations)
            
            timestamps = [face['timestamp'] for face in faces]
            if timestamps:
                first_seen = min(timestamps)
                last_seen = max(timestamps)
            else:
                first_seen = last_seen = "Unknown"
            
            report['cluster_details'][cluster_id] = {
                'face_count': len(faces),
                'locations_seen': dict(location_counts),
                'first_detected': first_seen,
                'last_detected': last_seen,
                'frequency_score': len(faces) / max(1, len(set(cluster_locations)))
            }
            
            # Generate recommendations
            if len(faces) >= 10:  # Frequently appearing unknown identity
                report['recommendations'].append({
                    'type': 'frequent_unknown',
                    'cluster_id': cluster_id,
                    'message': f"Cluster {cluster_id} appears frequently ({len(faces)} times). Consider adding to authorized database if legitimate.",
                    'priority': 'high'
                })
            
            if len(location_counts) == 1 and len(faces) >= 5:  # Always in same location
                location = list(location_counts.keys())[0]
                report['recommendations'].append({
                    'type': 'location_specific',
                    'cluster_id': cluster_id,
                    'message': f"Cluster {cluster_id} only appears at {location}. May be location-specific personnel.",
                    'priority': 'medium'
                })
        
        return report
        
    def load_authorized_faces(self):
        """Load pre-stored authorized face encodings from directory."""
        encodings_file = os.path.join(self.authorized_faces_dir, "face_encodings.pkl")
        
        if os.path.exists(encodings_file):
            # Load from pickle file
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.authorized_encodings = data['encodings']
                self.authorized_names = data['names']
            logger.info(f"Loaded {len(self.authorized_encodings)} authorized faces from cache")
        else:
            # Build from image files
            self._build_face_database()
            
        # Build FAISS index for fast similarity search
        if self.authorized_encodings:
            self._build_faiss_index()
    
    def _build_face_database(self):
        """Build face database from image files in authorized_faces directory."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(self.authorized_faces_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(self.authorized_faces_dir, filename)
                name = os.path.splitext(filename)[0]  # Use filename as person's name
                
                try:
                    # Load and encode face
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.authorized_encodings.append(face_encodings[0])
                        self.authorized_names.append(name)
                        logger.info(f"Added authorized face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        # Save encodings to cache
        if self.authorized_encodings:
            encodings_file = os.path.join(self.authorized_faces_dir, "face_encodings.pkl")
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.authorized_encodings,
                    'names': self.authorized_names
                }, f)
            logger.info(f"Saved {len(self.authorized_encodings)} face encodings to cache")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast face similarity search."""
        if not self.authorized_encodings:
            return
            
        # Convert to numpy array
        encodings_array = np.array(self.authorized_encodings, dtype=np.float32)
        
        # Build FAISS index
        dimension = encodings_array.shape[1]  # Should be 128 for face_recognition
        self.face_index = faiss.IndexFlatL2(dimension)
        self.face_index.add(encodings_array)
        
        logger.info(f"Built FAISS index with {len(self.authorized_encodings)} faces")
    
    def detect_and_verify_faces(self, frame: np.ndarray, location: str = "Unknown", 
                               timestamp: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect faces in frame, verify against authorized faces, and blur unauthorized ones.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, face_detections_info)
        """
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        face_info = []
        processed_frame = frame.copy()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if face matches any authorized face
            is_authorized, person_name, confidence = self._verify_face(face_encoding)
            
            face_data = {
                'location': (top, right, bottom, left),
                'is_authorized': is_authorized,
                'person_name': person_name,
                'confidence': confidence
            }
            face_info.append(face_data)
            
            if not is_authorized:
                # Blur unauthorized face
                processed_frame = self._blur_face_region(processed_frame, (top, right, bottom, left))
                logger.info(f"Blurred unauthorized face at location {(top, right, bottom, left)}")
            else:
                # Draw green rectangle for authorized faces (optional)
                cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"{person_name} ✓", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                logger.info(f"Authorized person detected: {person_name} (confidence: {confidence:.2f})")
        
        return processed_frame, face_info
    
    def _verify_face(self, face_encoding: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """
        Verify if a face encoding matches any authorized face.
        
        Args:
            face_encoding: Face encoding to verify
            
        Returns:
            Tuple of (is_authorized, person_name, confidence_score)
        """
        if not self.authorized_encodings:
            return False, None, 0.0
        
        if self.face_index:
            # Use FAISS for fast search
            query = np.array([face_encoding], dtype=np.float32)
            distances, indices = self.face_index.search(query, 1)
            
            distance = distances[0][0]
            best_match_idx = indices[0][0]
            
            # Convert L2 distance to similarity score
            similarity = 1 / (1 + distance)
            
            if similarity >= self.similarity_threshold:
                return True, self.authorized_names[best_match_idx], similarity
            else:
                return False, None, similarity
        else:
            # Fallback to face_recognition's compare_faces
            matches = face_recognition.compare_faces(self.authorized_encodings, face_encoding, 
                                                   tolerance=1-self.similarity_threshold)
            face_distances = face_recognition.face_distance(self.authorized_encodings, face_encoding)
            
            if any(matches):
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    confidence = 1 - face_distances[best_match_idx]
                    return True, self.authorized_names[best_match_idx], confidence
            
            return False, None, min(face_distances) if face_distances.size > 0 else 0.0
    
    def _blur_face_region(self, frame: np.ndarray, face_location: Tuple[int, int, int, int], 
                         blur_factor: int = 15) -> np.ndarray:
        """
        Blur a specific face region in the frame.
        
        Args:
            frame: Input frame
            face_location: (top, right, bottom, left) coordinates
            blur_factor: Blur intensity (higher = more blur)
            
        Returns:
            Frame with blurred face region
        """
        top, right, bottom, left = face_location
        
        # Extract face region
        face_region = frame[top:bottom, left:right]
        
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (blur_factor*2+1, blur_factor*2+1), 0)
        
        # Replace original face region with blurred version
        frame[top:bottom, left:right] = blurred_face
        
        return frame
    
    def add_authorized_face(self, image_path: str, person_name: str) -> bool:
        """
        Add a new authorized face to the database.
        
        Args:
            image_path: Path to the person's image
            person_name: Name/ID of the person
            
        Returns:
            True if face was successfully added, False otherwise
        """
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                logger.error(f"No face found in {image_path}")
                return False
            
            # Add to current session
            self.authorized_encodings.append(face_encodings[0])
            self.authorized_names.append(person_name)
            
            # Rebuild FAISS index
            self._build_faiss_index()
            
            # Save to cache
            encodings_file = os.path.join(self.authorized_faces_dir, "face_encodings.pkl")
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.authorized_encodings,
                    'names': self.authorized_names
                }, f)
            
            logger.info(f"Successfully added authorized face: {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding authorized face {person_name}: {e}")
            return False
    
    def get_face_match_summary(self, face_info_list: List[Dict]) -> Dict:
        """
        Generate summary of face matches for alert generation.
        
        Args:
            face_info_list: List of face detection info from detect_and_verify_faces
            
        Returns:
            Summary dictionary for use in alerts
        """
        total_faces = len(face_info_list)
        authorized_faces = sum(1 for face in face_info_list if face['is_authorized'])
        unauthorized_faces = total_faces - authorized_faces
        
        authorized_names = [face['person_name'] for face in face_info_list 
                          if face['is_authorized'] and face['person_name']]
        
        return {
            'total_faces': total_faces,
            'authorized_faces': authorized_faces,
            'unauthorized_faces': unauthorized_faces,
            'authorized_names': authorized_names,
            'has_unauthorized': unauthorized_faces > 0
        }