package com.translation_service.translation_service.user.application;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.translation_service.translation_service.common.model.User;
import com.translation_service.translation_service.user.domain.service.UserService;
import com.translation_service.translation_service.user.infras.jwt.JwtUtil;
import com.translation_service.translation_service.user.infras.persistence.MongoDBImpl;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Component;

import java.util.Optional;

@Component
public class UserApplicationService {
    private final UserService userService;
    private final MongoDBImpl mongoDBImpl;
    private final JwtUtil jwtUtil;
    private static final String SECRET_KEY = "YourSuperSecretKeyMustBeAtLeast32Bytes!";
    private static final Algorithm algorithm = Algorithm.HMAC256(SECRET_KEY);


    public UserApplicationService(UserService userService, MongoDBImpl mongoDBImpl, JwtUtil jwtUtil) {
        this.userService = userService;
        this.mongoDBImpl = mongoDBImpl;
        this.jwtUtil = jwtUtil;
    }

    public User getUserById(Integer id) {
        return Optional.ofNullable(userService.getUser(id))
                .orElseThrow(() -> new RuntimeException("User with ID " + id + " not found"));
    }

    public void addSentenceToUser(Long userId, String sentence) {
        System.out.println("Adding in Application layer ... ");
        userService.addSentenceToUser(userId, sentence);
    }

    public boolean validateToken(String token){
        String username = jwtUtil.extractUsername(token);
        DecodedJWT decodedJWT = JWT.require(algorithm)
                .build()
                .verify(token.replace("Bearer ", "")); // Remove "Bearer " prefix

        Long userId = decodedJWT.getClaim("userId").asLong();
        User foundUser = getUserById(userId.intValue());
        String password = foundUser.getPassword();

        return true;
    }


//    public void createUser(User user) {
////        UserDocument userDocument = UserMapper.toDocument(user);
////        mongoDBImpl.createUser(user.getId(), userDocument);  // This should save the user to MongoDB.
//    }



}
