package com.loginAndRegister.loginAndRegister_service.Util;

//import io.jsonwebtoken.Claims;
////import io.jsonwebtoken.Jwts;
//import io.jsonwebtoken.Jwts;
//import io.jsonwebtoken.SignatureAlgorithm;
//import io.jsonwebtoken.Jwts;
//import io.jsonwebtoken.SignatureAlgorithm;
//import io.jsonwebtoken.security.Keys;
//import io.jsonwebtoken.*;

import java.util.Date;

import com.auth0.jwt.algorithms.Algorithm;
import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import com.loginAndRegister.loginAndRegister_service.Service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import com.auth0.jwt.JWT;
import com.auth0.jwt.exceptions.JWTVerificationException;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.auth0.jwt.interfaces.JWTVerifier;

@Component
public class JwtUtil {

    private static final String SECRET_KEY = "YourSuperSecretKeyMustBeAtLeast32Bytes!";
    private static final Algorithm algorithm = Algorithm.HMAC256(SECRET_KEY);
//    @Autowired
    private final UserService userService;

    public JwtUtil(UserService userService) {
        this.userService = userService;
    }


    public String generateToken(String username) {
        User user = userService.findByUsername(username);
        return JWT.create()
                .withSubject(username)
                .withClaim("userId", user.getId())
                .withIssuedAt(new Date())
                .withExpiresAt(new Date(System.currentTimeMillis() + 1000 * 60 * 60 * 10)) // 10 hours
                .sign(algorithm);
    }

    public String extractUsername(String token) {
        try {
            return decodeToken(token).getSubject();
        } catch (JWTVerificationException e) {
            throw new IllegalArgumentException("Invalid token", e);
        }
    }

    public boolean validateToken(String token, String username) {
        try {
            DecodedJWT decodedJWT = decodeToken(token);
            return !isTokenExpired(token);
        } catch (JWTVerificationException e) {
            return false;
        }
    }

    private boolean isTokenExpired(String token) {
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