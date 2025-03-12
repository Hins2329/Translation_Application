package com.translation_service.translation_service.user.controller;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.exceptions.JWTVerificationException;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.translation_service.translation_service.user.application.UserApplicationService;
import com.translation_service.translation_service.common.model.User;
import com.translation_service.translation_service.user.infras.jwt.JwtUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/users")
public class UserController {
    @Autowired
    private MongoTemplate mongoTemplate;
    private final UserApplicationService userApplicationService;
    private static final String SECRET_KEY = "YourSuperSecretKeyMustBeAtLeast32Bytes";
    private final JwtUtil jwtUtil;
    private static final Algorithm algorithm = Algorithm.HMAC256(SECRET_KEY);

//    private final UserMapper userMapper;

    public UserController(MongoTemplate mongoTemplate, UserApplicationService userApplicationService, JwtUtil jwtUtil) {
        this.mongoTemplate = mongoTemplate;
        this.userApplicationService = userApplicationService;
//        this.userMapper = userMapper;
        this.jwtUtil = jwtUtil;
    }
    @GetMapping("/test-db")
    public ResponseEntity<String> testMongoConnection() {
        try {
            List<String> collections = mongoTemplate.getDb().listCollectionNames().into(new ArrayList<>());
            return ResponseEntity.ok("✅ MongoDB Connected! Collections: " + collections);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("❌ MongoDB Error: " + e.getMessage());
        }
    }
    @GetMapping("/test")
    public String test(){
        return "fuckyou";
    }

//    curl -X GET http://localhost:8444/users/9/
    @GetMapping("/{userId}")
    public ResponseEntity<User> getUser(@PathVariable Integer userId) {
        System.out.println("API hit: userId = " + userId); // Log incoming request

        Optional<User> user = Optional.ofNullable(userApplicationService.getUserById(userId));
        return user.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }

//    curl -X POST http://localhost:8444/users/9/sentences \
//            -H "Content-Type: application/json" \
//            -d '"yeah,fkyoutest2"'
//    @PostMapping("/sentences")
////    public ResponseEntity<String> addSentence(@PathVariable Long userId, @RequestBody String sentence) {
//    public ResponseEntity<String> addSentence(@RequestBody String sentence, @RequestHeader("Authorization") String token) {
//        DecodedJWT decodedJWT = JWT.require(algorithm).build().verify(token.replace("Bearer ", ""));
//        Long userId = decodedJWT.getClaim("userId").asLong();
//        userApplicationService.addSentenceToUser(userId, sentence);
//        return ResponseEntity.ok("Sentence added");
//    }
@PostMapping("/sentences")
//public ResponseEntity<String> addSentence(@RequestBody String sentence, @RequestHeader("Authorization") String token) {
public ResponseEntity<String> addSentence(@RequestBody Map<String, String> request,
                                          @RequestHeader("Authorization") String token) {
    try {
        System.out.println("im in Controller");

        // Verify and decode JWT token
        DecodedJWT decodedJWT = JWT.require(algorithm)
                .build()
                .verify(token.replace("Bearer ", "")); // Remove "Bearer " prefix

        Long userIdFromToken = decodedJWT.getClaim("userId").asLong();
        String sentence = request.get("sentence"); // Extract sentence from JSON

        if (sentence == null || sentence.trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Sentence cannot be empty");
        }
        // Check if the userId in the request matches the token
//        Long userIdFromRequest = Long.parseLong(request.get("userId"));

        userApplicationService.addSentenceToUser(userIdFromToken, sentence);
        return ResponseEntity.ok("Sentence added");
    } catch (JWTVerificationException e) {
        System.err.println("Invalid token: " + e.getMessage());
        return ResponseEntity.status(401).body("Invalid token");
    }catch (Exception e) {
        System.err.println("Unexpected error: " + e.getMessage());
        return ResponseEntity.status(500).body("Internal server error");
    }
}

    @GetMapping("/verify-token")
    public ResponseEntity<?> verifyToken(@CookieValue(value = "token", required = false) String token) {
//        System.out.println("i am in translate-service verify token");
//        if (token == null || !userApplicationService.validateToken(token)) {
//            System.out.println("verify token failed");
//            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Invalid token");
//        }
        return ResponseEntity.ok("Token is valid");
    }


}