package com.translation_service.translation_service.user.infras.jwt;


import java.util.Date;

import com.auth0.jwt.algorithms.Algorithm;
//import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
//import com.loginAndRegister.loginAndRegister_service.Service.UserService;
import com.translation_service.translation_service.common.model.User;
import com.translation_service.translation_service.user.domain.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import com.auth0.jwt.JWT;
import com.auth0.jwt.exceptions.JWTVerificationException;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.auth0.jwt.interfaces.JWTVerifier;

@Component
public class JwtUtil {
//    private final String SECRET_KEY = "SecretKeyThisShouldBeLongEnoughForHS256"; // Should be 256 bits or more
//    private static final String SECRET_KEY = "YourSuperSecretKeyMustBeAtLeast32Bytes!";
//    private final Key secretKey = Keys.secretKeyFor(SignatureAlgorithm.HS256);

    private static final String SECRET_KEY = "YourSuperSecretKeyMustBeAtLeast32Bytes!";
    private static final Algorithm algorithm = Algorithm.HMAC256(SECRET_KEY);
//    @Autowired
//    private final UserService userService;
//
//    public JwtUtil(UserService userService) {
//        this.userService = userService;
//    }
//
//
//    public String generateToken(String username) {
//        User user = userService.findByUsername(username);
//        return JWT.create()
//                .withSubject(username)
//                .withClaim("userId", user.getId())  // Store userId as a claim
//                .withIssuedAt(new Date())
//                .withExpiresAt(new Date(System.currentTimeMillis() + 1000 * 60 * 60 * 10)) // 10 hours
//                .sign(algorithm);
//    }

    public String extractUsername(String token) {
        System.out.println("im in extractUsername");
        try {
            return decodeToken(token).getSubject();
        } catch (JWTVerificationException e) {
            throw new IllegalArgumentException("Invalid token", e);
        }
    }

    public boolean validateToken(String token, String username) {
        System.out.println("im in validateToken");
        try {
            DecodedJWT decodedJWT = decodeToken(token);
            return !isTokenExpired(token);
        } catch (JWTVerificationException e) {
            return false;
        }
    }


    private boolean isTokenExpired(String token) {
        System.out.println("im in isTokenExpired");
        try {
            return decodeToken(token).getExpiresAt().before(new Date());
        } catch (JWTVerificationException e) {
            return true; // If decoding fails, consider it expired
        }
    }
    private DecodedJWT decodeToken(String token) {
        JWTVerifier verifier = JWT.require(algorithm).build();
        return verifier.verify(token);
    }
}